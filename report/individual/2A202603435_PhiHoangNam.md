# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- 2A202603435
- Phí Hoàng Nam
- 8276

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | Có |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | Có |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | Có |

## 2. Kết quả chính

| Mô hình | F1 test | F1 SUB 40 | p95 ms/tin | USD/1k tin trong notebook |
| --- | ---: | ---: | ---: | ---: |
| BoW + LogReg | 0.880 | 0.919 | 0.056 | 0 |
| BoW + MLP | 0.954 | 0.947 | 0.426 | 0 |
| MiniLM + kNN | 0.910 | 0.927 | 40.032 | 0 |
| MiniLM + LogReg | 0.941 | 0.974 | 32.275 | 0 |
| MiniLM + MLP head | 0.950 | 0.974 | 37.359 | 0 |
| LLM nhỏ zero-shot | Chỉ đo SUB | 0.905 | 2733.328 | 0* |
| LLM lớn zero-shot | Chỉ đo SUB | 0.927 | 2798.166 | 0* |
| LLM nhỏ few-shot | Chỉ đo SUB | 0.950 | 3037.491 | 0* |

LLM sử dụng Groq với openai/gpt-oss-20b và openai/gpt-oss-120b. Dấu * cho biết notebook chưa khai báo giá hai model này nên chi phí mặc định bằng 0; chưa thể kết luận API miễn phí. Các mô hình cục bộ không có phí API nhưng vẫn có chi phí máy tính.

BoW + MLP đạt F1 cao nhất trên toàn bộ test, bằng 0.954. Hai phương án MiniLM + LogReg và MiniLM + MLP head cùng đạt F1 SUB 0.974. SUB chỉ có 40 tin nên các chênh lệch nhỏ chưa đủ để khẳng định mô hình nào tốt hơn chắc chắn.

## 3. Dự đoán và thực tế

Dự đoán về bỏ ReLU: Tôi dự đoán mạng không có ReLU sẽ có F1 giảm hoặc gần LogReg Phần 1. Thực tế, mạng không ReLU đạt F1 0.954, bằng MLP có ReLU và cao hơn LogReg Phần 1 đạt 0.880. Hai lớp Linear liên tiếp tương đương một phép biến đổi affine, nhưng cách khởi tạo, tối ưu và huấn luyện khác nhau nên kết quả thực nghiệm không nhất thiết giống nhau.

Dự đoán về bộ test khó: Tôi dự đoán accuracy giảm mạnh, BoW kém hơn Embedding và LLM. Thực tế, BoW chỉ đạt 50–54%, Embedding đạt 68–74%, trong khi LLM zero-shot đạt 88% và LLM có prompt phòng thủ đạt 92%. Dự đoán đúng về xu hướng tương đối, nhưng LLM giữ kết quả tốt hơn mức tôi hình dung.

Dự đoán về BoW + LogReg: Tôi dự đoán F1 khoảng 0.85 và precision cao hơn recall. Thực tế F1 đạt 0.880, precision bằng 1.000 và recall bằng 0.786. F1 cao hơn dự đoán 0.030; mô hình nhận diện chắc chắn một số tín hiệu spam nhưng vẫn bỏ sót nhiều tin spam.

## 4. Phân tích 50 case khó

BoW + LogReg tụt nhiều nhất: F1 giảm từ 0.919 trên SUB xuống 0.286 trên bộ khó, giảm 0.633. Mô hình này chỉ đoán đúng 0% nhóm spam giọng tự nhiên, khoảng 12,5% nhóm thao túng/lừa đảo và 16,7% nhóm né từ khóa.

Nguyên nhân là BoW phụ thuộc các từ xuất hiện trong từ điển, nên khó nhận biết spam viết giống lời nhắn thông thường hoặc cố tình thay đổi từ khóa.

Hai case sai được thể hiện rõ trong output của LLM có prompt phòng thủ:
- Case #2: “FREE pizza in the office kitchen right now, come quick before it's gone”. Đây là ham nhưng bị đoán spam. Model có thể bị ảnh hưởng bởi từ “FREE” và cách thúc giục, chưa nhận ra ngữ cảnh thông báo nội bộ.
- Case #47: Tin kể lại việc nhận thông báo trúng thưởng và chê trò lừa đảo. Đây là ham nhưng bị đoán spam. Model có thể tập trung vào nội dung trúng thưởng được nhắc lại thay vì ý định của cả tin nhắn.
Prompt phòng thủ giúp accuracy tăng từ 88% lên 92%, F1 tăng từ 0.903 lên 0.935. Tuy nhiên nhóm “nói về spam” giảm từ 100% xuống 75%, cho thấy phòng thủ vẫn có thể làm tăng chặn nhầm.

## 5. Thử thách cá nhân

Tôi nhận đề #3 theo SEED 8276: thay số nơ-ron lớp ẩn MLP thành 4 và 512, so sánh số tham số, F1 test và thời gian huấn luyện.

Trong notebook hiện tại, tôi đã thử mạng gồm bốn lớp ẩn, mỗi lớp 512 nơ-ron. Kết quả:
Cấu hình đã chạy	Số tham số	Thời gian train	F1 test
Bốn lớp ẩn, mỗi lớp 512 nơ-ron	1.812.993	15,29 giây	0.949


Cấu hình này khác yêu cầu so sánh một lớp ẩn có 4 và 512 nơ-ron, nên phần thử thách chưa hoàn thành đầy đủ. Kết quả hiện có chưa chứng minh mạng lớn tốt hơn: F1 0.949 vẫn thấp hơn MLP cơ bản đạt 0.954. Mạng nhỏ phù hợp khi tài nguyên hạn chế và mô hình lớn không cải thiện đủ chất lượng trên validation.

## 6. Phương án cho sản phẩm thật

Với 10 triệu SMS/ngày, hệ thống cần xử lý trung bình khoảng 116 tin/giây.

Tôi chọn BoW + MLP làm phương án cơ sở để thử nghiệm, vì F1 test đạt 0.954, p95 chỉ 0.426 ms/tin và throughput đo cả lô khoảng 41.238 tin/giây. Tuy nhiên accuracy trên bộ khó chỉ đạt 54%, nên cần bổ sung cơ chế xử lý các tin khó và kiểm soát chặn nhầm.

MiniLM + LogReg đạt p95 32.275 ms nhưng throughput chỉ khoảng 42 tin/giây trong phép đo hiện tại. Do đó, đạt p95 dưới 50 ms chưa đồng nghĩa một tiến trình đủ xử lý 116 tin/giây; cần tăng công suất và đo lại dưới tải thực tế.

Tôi đề xuất dùng mô hình nhanh xử lý các tin chắc chắn, chuyển tin chưa chắc chắn sang mô hình ngữ nghĩa. LLM có thể hỗ trợ phân tích bất đồng bộ vì p95 hiện khoảng 2,7–3 giây, không phù hợp với đường xử lý bắt buộc hoàn tất trong 50 ms.

Đánh đổi chính là tốc độ của BoW với khả năng xử lý tin khó của Embedding/LLM. Chi phí LLM thực tế chưa xác định do notebook thiếu giá Groq; không thể dùng số 0 đang hiển thị để kết luận chi phí vận hành bằng 0.

## 7. AI log và điều đã học

| Phần | Tôi hỏi AI gì | AI hỗ trợ gì | Tôi kiểm chứng gì |
| --- | --- | --- | --- |
| Phần 1 & 2 — ML và DL | Nhờ giải thích Gradient Descent, hàm mất mát BCE và cơ chế cập nhật trọng số trong Logistic Regression, MLP | Giải thích công thức, lan truyền ngược và cập nhật trọng số ngược chiều gradient | Tự viết các hàm bằng NumPy, PyTorch và chạy các unit test có sẵn |
| Phần 3 — Embedding | Hỏi cách tối ưu tìm kiếm cosine bằng phép nhân ma trận `M @ q`, tránh vòng lặp qua từng dòng | Hướng dẫn chuẩn hóa vector, nhân ma trận và dùng `argsort` lấy top-k | Tự viết `cosine_similarity`, `top_k_similar`; kiểm tra thứ tự và chiều sắp xếp giảm dần |
| Phần 4 & 5 — LLM và tổng hợp | Nhờ gợi ý phân tích 50 case khó, thiết kế router nhiều tầng và chọn mô hình cho sản phẩm | Đề xuất BoW → Encoder + LogReg → LLM fallback; hướng dẫn xem xét p95 | Cô đọng thành ý của mình và chạy notebook để đối chiếu kết quả |
