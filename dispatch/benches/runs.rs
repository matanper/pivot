//! Equal-key runs across aggregation signatures and key representations.
use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{ArrayRef, Float64Array, Int32Array, Int64Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field, Schema};
use criterion::{BatchSize, Criterion, Throughput, black_box};
use dispatch::{
    AggregationKind as Kind, AggregationSlot as Slot, Compiled, CountSlot, CountValidSlot,
    Dispatch, Dynamic, IntKeyExtractor, MaxSlot, MinSlot, RecordBatchOperatorSpec, SumSlot,
    memory_ctx, values_input,
};
use std::{sync::Arc, time::Duration};

fn input(rows: usize, shape: &str) -> Vec<RecordBatch> {
    let run = match shape {
        "run_16" => 16,
        "run_64" => 64,
        _ => 1,
    };
    let mut keys: Vec<i64> = (0..rows).map(|row| ((row / run) % 8192) as i64).collect();
    if shape == "shuffled" {
        let mut seed = 42u64;
        for index in (1..rows).rev() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            keys.swap(index, seed as usize % (index + 1));
        }
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("value", DataType::Int32, false),
        Field::new("wide", DataType::Int64, false),
        Field::new("nullable", DataType::Int32, true),
        Field::new("float", DataType::Float64, false),
        Field::new("text", DataType::Utf8View, false),
    ]));
    keys.chunks(65536)
        .map(|keys| {
            let n = keys.len();
            let columns: Vec<ArrayRef> = vec![
                Arc::new(Int64Array::from(keys.to_vec())),
                Arc::new(Int32Array::from_iter_values(
                    (0..n).map(|row| (row % 101) as i32 - 50),
                )),
                Arc::new(Int64Array::from_iter_values(
                    (0..n).map(|row| i64::MAX / 1000 + row as i64),
                )),
                Arc::new(Int32Array::from(
                    (0..n)
                        .map(|row| (row % 7 != 0).then_some((row % 101) as i32 - 50))
                        .collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from_iter_values(
                    (0..n).map(|row| (row % 101) as f64 * 0.5),
                )),
                Arc::new(StringViewArray::from_iter_values(
                    (0..n).map(|row| format!("a-long-string-value-{:04}", row % 101)),
                )),
            ];
            RecordBatch::try_new(schema.clone(), columns).unwrap()
        })
        .collect()
}
fn slot(kind: Kind, column: usize, output: DataType) -> Slot {
    Slot::new(kind, column, output)
}
fn build(input: RecordBatchOperatorSpec, signature: &str) -> RecordBatchOperatorSpec {
    let count = slot(Kind::CountStar, 0, DataType::Int64);
    let sum = slot(Kind::Sum, 1, DataType::Int64);
    match signature {
        "sum" => input.group_by_aggregate::<IntKeyExtractor<Int64Type>,Compiled<(SumSlot<Int32Type>,)>>(vec![0],vec![sum],None,()),
        "wide_sum" => input.group_by_aggregate::<IntKeyExtractor<Int64Type>,Compiled<(SumSlot<Int64Type,i128>,)>>(vec![0],vec![slot(Kind::Sum,2,DataType::Decimal128(38,0))],None,()),
        "extrema" => input.group_by_aggregate::<IntKeyExtractor<Int64Type>,Compiled<(MinSlot<Int32Type>,MaxSlot<Int32Type>)>>(vec![0],vec![slot(Kind::Min,1,DataType::Int64),slot(Kind::Max,1,DataType::Int64)],None,()),
        "count_sum" => input.group_by_aggregate::<IntKeyExtractor<Int64Type>,Compiled<(CountSlot,SumSlot<Int32Type>)>>(vec![0],vec![count,sum],None,()),
        "nullable" => input.group_by_aggregate::<IntKeyExtractor<Int64Type>,Compiled<(CountValidSlot,SumSlot<Int32Type>,MinSlot<Int32Type>,MaxSlot<Int32Type>),u8>>(vec![0],vec![slot(Kind::Count,3,DataType::Int64),slot(Kind::Sum,3,DataType::Int64),slot(Kind::Min,3,DataType::Int64),slot(Kind::Max,3,DataType::Int64)],None,()),
        "dynamic_numeric" => input.group_by_aggregate::<IntKeyExtractor<Int64Type>,Dynamic<i128>>(vec![0],vec![count,sum,slot(Kind::Sum,4,DataType::Float64),slot(Kind::Min,4,DataType::Float64),slot(Kind::Max,4,DataType::Float64)],None,()),
        "dynamic_strings" => input.group_by_aggregate::<IntKeyExtractor<Int64Type>,Dynamic<i128>>(vec![0],vec![count,slot(Kind::Min,5,DataType::Utf8View),slot(Kind::Max,5,DataType::Utf8View)],None,()),
        _ => unreachable!(),
    }
}
fn key_input(rows: usize, shape: &str) -> Vec<RecordBatch> {
    input(rows, shape)
        .into_iter()
        .map(|batch| {
            let keys = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let mut fields = batch
                .schema()
                .fields()
                .iter()
                .map(|field| field.as_ref().clone())
                .collect::<Vec<_>>();
            let mut columns = batch.columns().to_vec();
            let extra: Vec<ArrayRef> = vec![
                Arc::new(StringViewArray::from_iter_values(
                    keys.values()
                        .iter()
                        .map(|key| format!("same-prefix-long-ü-key-{key:04}")),
                )),
                Arc::new(StringViewArray::from_iter_values(
                    keys.values().iter().map(|key| format!("k{key:04}")),
                )),
                Arc::new(Int64Array::from_iter_values(
                    keys.values().iter().map(|key| key / 16),
                )),
                Arc::new(Int32Array::from_iter_values(
                    keys.values().iter().map(|key| (key % 16) as i32),
                )),
                Arc::new(Int64Array::from(
                    keys.values()
                        .iter()
                        .map(|key| (key % 5 != 0).then_some(key / 16))
                        .collect::<Vec<_>>(),
                )),
                Arc::new(StringViewArray::from_iter(keys.values().iter().map(
                    |key| (key % 7 != 0).then(|| format!("nullable-long-key-{key:04}")),
                ))),
                Arc::new(
                    arrow_array::Decimal128Array::from_iter_values(
                        keys.values().iter().map(|key| *key as i128 * 100),
                    )
                    .with_precision_and_scale(20, 2)
                    .unwrap(),
                ),
            ];
            for column in extra {
                fields.push(Field::new(
                    format!("key{}", columns.len()),
                    column.data_type().clone(),
                    column.null_count() > 0,
                ));
                columns.push(column);
            }
            RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
        })
        .collect()
}
fn build_keys(input: RecordBatchOperatorSpec, representation: &str) -> RecordBatchOperatorSpec {
    use dispatch::{
        HashOnlyIntKeyExtractor, IntPairKeyExtractor, IntStrKeyExtractor, RowKeyExtractor,
        RowKeySchema, StringKeyExtractor,
    };
    type Value = Compiled<(CountSlot, SumSlot<Int32Type>)>;
    let slots = vec![
        slot(Kind::CountStar, 0, DataType::Int64),
        slot(Kind::Sum, 1, DataType::Int64),
    ];
    match representation {
        "integer" => {
            input.group_by_aggregate::<IntKeyExtractor<Int64Type>, Value>(vec![0], slots, None, ())
        }
        "long_string" => {
            input.group_by_aggregate::<StringKeyExtractor, Value>(vec![6], slots, None, ())
        }
        "short_string" => {
            input.group_by_aggregate::<StringKeyExtractor, Value>(vec![7], slots, None, ())
        }
        "integer_pair" => input
            .group_by_aggregate::<IntPairKeyExtractor<Int64Type, Int32Type>, Value>(
                vec![8, 9],
                slots,
                None,
                (),
            ),
        "integer_string" => input.group_by_aggregate::<IntStrKeyExtractor<Int64Type>, Value>(
            vec![8, 6],
            slots,
            None,
            (),
        ),
        "row_triple" => input.group_by_aggregate::<RowKeyExtractor, Value>(
            vec![8, 9, 6],
            slots,
            None,
            RowKeySchema::new(
                vec![DataType::Int64, DataType::Int32, DataType::Utf8View],
                vec![false; 3],
            ),
        ),
        "nullable_row" => input.group_by_aggregate::<RowKeyExtractor, Value>(
            vec![10, 11],
            slots,
            None,
            RowKeySchema::new(vec![DataType::Int64, DataType::Utf8View], vec![true; 2]),
        ),
        "decimal_row" => input.group_by_aggregate::<RowKeyExtractor, Value>(
            vec![12],
            slots,
            None,
            RowKeySchema::new(vec![DataType::Decimal128(20, 2)], vec![false]),
        ),
        "hash_only_distinct" => {
            input.group_by_distinct_count::<HashOnlyIntKeyExtractor<Int64Type>>(vec![0])
        }
        _ => unreachable!(),
    }
}

fn main() {
    let rows = std::env::var("PIVOT_BENCH_ROWS")
        .map(|v| v.parse().unwrap())
        .unwrap_or(1048576);
    let engine = Dispatch::spin_up(2, 768, None);
    let dispatcher = engine.dispatcher().clone();
    let mut criterion = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(1))
        .sample_size(20)
        .configure_from_args();
    let mut group = criterion.benchmark_group("run_values");
    group.throughput(Throughput::Elements(rows as u64));
    for shape in ["run_1", "run_16", "run_64", "shuffled"] {
        let batches = input(rows, shape);
        for signature in [
            "sum",
            "wide_sum",
            "extrema",
            "count_sum",
            "nullable",
            "dynamic_numeric",
            "dynamic_strings",
        ] {
            group.bench_function(format!("{signature}/{shape}"), |b| {
                b.iter_batched(
                    || dispatcher.run_on_workers(|| memory_ctx().zero_dirty_buffers()),
                    |_| {
                        let source = values_input(&dispatcher, batches.clone()).record_batches();
                        black_box(build(source, signature).collect().unwrap());
                    },
                    BatchSize::PerIteration,
                )
            });
        }
    }
    group.finish();
    let mut key_group = criterion.benchmark_group("run_keys");
    key_group.throughput(Throughput::Elements(rows as u64));
    for shape in ["run_1", "run_16", "run_64", "shuffled"] {
        let batches = key_input(rows, shape);
        for representation in [
            "integer",
            "long_string",
            "short_string",
            "integer_pair",
            "integer_string",
            "row_triple",
            "nullable_row",
            "decimal_row",
            "hash_only_distinct",
        ] {
            key_group.bench_function(format!("{representation}/{shape}"), |b| {
                b.iter_batched(
                    || dispatcher.run_on_workers(|| memory_ctx().zero_dirty_buffers()),
                    |_| {
                        let source = values_input(&dispatcher, batches.clone()).record_batches();
                        black_box(build_keys(source, representation).collect().unwrap());
                    },
                    BatchSize::PerIteration,
                )
            });
        }
    }
    key_group.finish();
    criterion.final_summary();
    engine.exit();
}
