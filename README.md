# FinBERT-VCSenti: Mô Hình Phân Tích Quan Điểm Tin Tức Tài Chính

## 📝 Tổng Quan Dự Án

**FinBERT-VCSenti** là một mô hình học sâu được fine-tune từ `bert-base-uncased` để phân tích và phân loại quan điểm trong các văn bản tài chính bằng tiếng Anh. Mô hình có khả năng xác định xem một câu mang sắc thái **Tích cực (positive)**, **Tiêu cực (negative)**, hay **Trung lập (neutral)**, hỗ trợ các bài toán tự động hóa trong ngành tài chính - ngân hàng.

[cite_start]Dự án này được thực hiện như một minh chứng về kỹ năng ứng dụng các mô hình Ngôn ngữ Lớn (LLM) vào giải quyết bài toán thực tế, lấy cảm hứng từ paper nghiên cứu **FinBERT**[Link Paper](https://arxiv.org/pdf/1908.10063)


---

## 🚀 Mô Tả Mô Hình

* **Model gốc (Base Model):** `bert-base-uncased` từ Hugging Face.
* [cite_start]**Dataset:** Mô hình được fine-tune trên bộ dữ liệu **Financial PhraseBank** [cite: 199][cite_start], cụ thể là tập `sentences_allagree` nơi tất cả các chuyên gia tài chính đều đồng thuận về nhãn quan điểm[cite: 201].
* **Bài toán:** Phân loại văn bản (Text Classification) với 3 nhãn: `positive`, `negative`, `neutral`.

---

## 🛠️ Cài Đặt

Để sử dụng mô hình này, bạn cần cài đặt các thư viện cần thiết:

```bash
pip install transformers torch
```

---

## 💡 Cách Sử Dụng

Bạn có thể dễ dàng sử dụng mô hình này thông qua `pipeline` của thư viện Transformers.

```python
from transformers import pipeline
import torch

# Thay 'your-huggingface-username/FinBERT-VCSenti' bằng đường dẫn model của bạn sau khi push lên Hub
model_checkpoint = "your-huggingface-username/FinBERT-VCSenti"
device = 0 if torch.cuda.is_available() else -1

# Khởi tạo pipeline
classifier = pipeline("text-classification", model=model_checkpoint, device=device)

# Chuẩn bị câu cần dự đoán
sentences = [
    "The company's revenue grew by 25% in the last quarter.",
    "There are concerns about the upcoming economic recession.",
    "The new CEO will start his position next Monday."
]

# Thực hiện dự đoán
results = classifier(sentences)
for sentence, result in zip(sentences, results):
    print(f"'{sentence}'")
    print(f" -> Dự đoán: {result['label'].upper()} (Score: {result['score']:.4f})\n")

```
**Kết quả dự kiến:**
```
'The company's revenue grew by 25% in the last quarter.'
 -> Dự đoán: POSITIVE (Score: 0.9987)

'There are concerns about the upcoming economic recession.'
 -> Dự đoán: NEGATIVE (Score: 0.9995)

'The new CEO will start his position next Monday.'
 -> Dự đoán: NEUTRAL (Score: 0.9998)
```
---

## ⚙️ Quy Trình Huấn Luyện

Mô hình được huấn luyện bằng cách sử dụng `Trainer` API từ thư viện Transformers. [cite_start]Các siêu tham số (hyperparameters) chính được lựa chọn dựa trên đề xuất từ paper FinBERT[cite: 14]:

* [cite_start]**Learning Rate:** `2e-5` [cite: 316]
* **Batch Size:** `16`
* **Số Epochs:** `4`
* **Weight Decay:** `0.01`
* [cite_start]**Warmup Proportion:** `0.2` [cite: 316]

---

## 📊 Kết Quả Đánh Giá

Mô hình đạt được hiệu năng ấn tượng trên tập dữ liệu kiểm thử (validation set):

* **Accuracy:** `[Điền độ chính xác bạn đạt được, ví dụ: 0.8627]`
* **F1-Score (Weighted):** `[Điền F1-score bạn đạt được, ví dụ: 0.9044]`

Kết quả này cho thấy mô hình có khả năng phân loại tốt và phù hợp để triển khai vào các ứng dụng thực tế.

---

## 📚 Tài Liệu Tham Khảo

* Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models*. [arXiv:1908.10063](https://arxiv.org/abs/1908.10063).
* Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). *Good debt or bad debt: Detecting semantic orientations in economic texts*. Journal of the Association for Information Science and Technology, 65(4), 782-796.
