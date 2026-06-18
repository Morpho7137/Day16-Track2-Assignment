# Short Report for Section 7.8

I used the CPU + LightGBM fallback approach on AWS because the original GPU path was restricted by account quota and approval limits, and the benchmark was later completed successfully on an `r5.2xlarge` instance.
The dataset used was `mlg-ulb/creditcardfraud`, which contains 284,807 rows and 31 columns, making it suitable for a practical fraud detection benchmark.
The data loading time was `1.618363` seconds, while the total training time was only `1.451148` seconds, showing that the CPU instance was strong enough for gradient boosting on this dataset.
The model achieved `AUC-ROC = 0.975399` and `Accuracy = 0.994909`, which indicates strong overall classification performance.
The `Recall = 0.897959` value is relatively high, meaning the model detected most fraudulent transactions, although `Precision = 0.239130` and `F1-score = 0.377682` show that false positives were still present.
For inference performance, the latency for a single row was `0.969592 ms`, and the throughput for a batch of 1000 rows reached `669118.763895 rows/sec`, which is very fast for CPU inference.
Compared with the GPU option, this CPU-based solution is not intended for LLM serving, but it is appropriate for tabular machine learning because LightGBM does not require GPU acceleration to perform well.
The reason for using CPU instead of GPU was that GPU quota on AWS was initially limited or difficult to approve, while `r5.2xlarge` was available and sufficient to meet the lab objectives for training, inference, and benchmarking.
