# Bao cao ngan Section 7.8

Toi su dung phuong an CPU + LightGBM tren AWS do ban dau tai khoan gap han che voi huong GPU, sau do benchmark duoc chay thanh cong tren instance `r5.2xlarge`.
Bo du lieu su dung la `mlg-ulb/creditcardfraud` voi 284,807 dong du lieu va 31 cot, phu hop de danh gia bai toan phat hien gian lan giao dich.
Thoi gian load data dat `1.618363` giay, trong khi thoi gian training chi `1.451148` giay, cho thay CPU du manh de xu ly gradient boosting nhanh voi bo du lieu nay.
Mo hinh dat `AUC-ROC = 0.975399` va `Accuracy = 0.994909`, cho thay kha nang phan biet mau gian lan va binh thuong o muc tot.
Chi so `Recall = 0.897959` kha cao, nghia la mo hinh bat duoc phan lon giao dich gian lan, nhung `Precision = 0.239130` va `F1-score = 0.377682` cho thay van con nhieu du doan duong tinh gia.
Ve suy luan, do tre cho 1 dong du lieu la `0.969592 ms`, va throughput cho lo 1000 dong dat `669118.763895 rows/sec`, the hien toc do inference rat nhanh tren CPU.
So voi phuong an GPU, cach lam nay khong phu hop cho LLM inference nhung lai hop ly voi bai toan tabular machine learning, vi LightGBM khong can GPU de dat hieu nang tot.
Ly do phai dung CPU thay GPU la yeu cau quota GPU tren AWS de bi gioi han hoac kho duyet, trong khi `r5.2xlarge` co san va dap ung duoc muc tieu training, inference, va benchmarking cua bai lab.
