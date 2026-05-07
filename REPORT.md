# Lab 21 — Evaluation Report: LoRA/QLoRA Fine-tuning

**Học viên**: Hồ Đắc Toàn — 2A202600057  
**Ngày nộp**: 2026-05-07  
**Submission option**: A (Lightweight ZIP)  
**Module**: AICB-P2T3 · Ngày 21 · Chương 5 — Fine-tuning & An Toàn

---

## 1. Setup

| Hạng mục | Chi tiết |
|----------|---------|
| **Base model** | `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` |
| **Quantization** | 4-bit NF4 (QLoRA) |
| **Dataset** | `iamtarun/python_code_instructions_18k_alpaca` |
| **Dataset size** | 500 samples → 450 train + 50 eval (90/10 split, seed=42) |
| **Domain** | Python coding instructions |
| **max_seq_length** | 512 (p95 của dataset, capped để tiết kiệm VRAM trên T4) |
| **GPU** | Tesla T4 — 15.6 GB VRAM · CUDA 12.8 · PyTorch 2.10.0 |
| **Training cost** | ~$0.14 (24.4 phút tổng @ $0.35/hr T4 rate) |
| **Target modules** | ALL layers: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| **Hyperparameters** | 3 epochs · lr=2e-4 · cosine schedule · warmup=10% · effective batch=8 · optimizer=adamw_8bit |

**Lý do chọn dataset Python coding**: Dataset có cấu trúc instruction/input/output rõ ràng, output là code Python với độ dài vừa phải, phù hợp để fine-tune trên T4 với max_seq_length=512. Domain coding cũng dễ đánh giá qualitative (code đúng hay sai).

---

## 2. Rank Experiment Results

| Rank | Trainable Params | % Total Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-----------------|----------------|------------|-----------|-----------|------------|
| **r=8** | 12,156,928 | 0.38% | 7.66 phút | 5.74 GB | 0.7935 | **2.211** |
| **r=16** | 24,313,856 | 0.75% | 8.44 phút | 5.07 GB | 0.8179 | 2.266 |
| **r=64** | 97,255,424 | 2.94% | 8.26 phút | 7.41 GB | 0.9086 | 2.481 |
| Base | — | — | — | — | — | — |

> **Lưu ý về alpha**: Tỉ lệ alpha/r = 2 được giữ nhất quán cho cả 3 experiments (r=8 → alpha=16, r=16 → alpha=32, r=64 → alpha=128), đảm bảo so sánh công bằng về scaling.

**Quan sát chính**:

- **Perplexity nghịch chiều với rank** — r=8 cho kết quả tốt nhất (ppl=2.211), r=64 tệ nhất (ppl=2.481). Đây là kết quả ngược trực giác so với lý thuyết (rank cao hơn = nhiều tham số hơn = lẽ ra học được nhiều hơn).
- **Training time gần như bằng nhau** — 7.66 đến 8.44 phút cho cả 3 ranks, cho thấy với dataset nhỏ (450 samples), bottleneck không phải số tham số mà là data throughput.
- **VRAM tăng đáng kể ở r=64** — 7.41 GB (r=64) so với 5.07–5.74 GB (r=8, r=16), tăng ~46% VRAM nhưng perplexity lại tệ hơn.

---

## 3. Loss Curve Analysis

> **Lưu ý T4-specific**: Eval-during-training được tắt (`eval_strategy="no"`) để tránh OOM trên T4 16GB. Do đó chỉ có training loss curve, không có eval loss curve real-time.

**Training loss r=16** giảm đều từ ~1.6 (step đầu) xuống ~0.85 (step 171) qua 3 epochs với cosine LR schedule. Không có dấu hiệu loss bị spike hay plateau bất thường.

**Nhận xét về overfitting**:

Mặc dù không có eval curve trong khi train, kết quả eval sau training cho thấy:

- **r=8**: eval_loss=0.7935 < training loss cuối (~0.85) → **không overfit**, model đang generalize tốt.
- **r=16**: eval_loss=0.8179 ≈ training loss cuối → **borderline**, gần như balance.
- **r=64**: eval_loss=0.9086 > training loss cuối → **dấu hiệu overfit nhẹ** — eval loss cao hơn training loss, có nghĩa là model với rank cao đang memorize training data thay vì generalize.

**Giải thích**: Dataset chỉ có 500 samples (450 train). Với r=64 có 97M trainable params trên 450 examples, tỉ lệ params-per-sample quá cao (~216K params/sample), dẫn đến overfit. r=8 với 12M params (~27K params/sample) phù hợp hơn với dataset nhỏ này.

---

## 4. Qualitative Comparison (5 Examples)

So sánh base model vs fine-tuned (r=16) trên 5 test prompts Python domain:

### Example 1 — Đảo ngược chuỗi

**Prompt**: Viết một hàm Python để đảo ngược một chuỗi mà không dùng các hàm có sẵn.

**Base model**:
```python
def reverse_string(string):
    result = ""
    for i in range(len(string)-1, -1, -1):
        result += string[i]
    return result

if __name__ == '__main__':
    string = "Hello World!"
    print(reverse_string(string))  # Output: "!dlroW olleH"
```

**Fine-tuned (r=16)**:
```python
def reverse_string(str):
    result = ""
    for i in range(len(str) - 1, -1, -1):
        result += str[i]
    return result

# test
print(reverse_string("Hello World"))  # Output: "dlroW olleH"
```

**Nhận xét**: Cả hai đều đúng về logic. Fine-tuned thêm comment test inline theo style của training data (Python coding dataset). Micro-improvement về format.

---

### Example 2 — Đọc CSV với pandas

**Prompt**: Làm cách nào để đọc một file CSV và in ra 5 dòng đầu tiên bằng thư viện pandas?

**Base model**:
```python
import pandas as pd
  
# đọc file csv
df = pd.read_csv('file.csv')
 
# in ra 5 dòng đầu tiên
for index, row in df.head(5).iterrows():
    print(row)
```

**Fine-tuned (r=16)**:
```python
import pandas as pd

data = pd.read_csv('file.csv')

for index, row in data.head(5).iterrows():
    print(row)
    # Output: row1, row2...
```

**Nhận xét**: Base model thêm comments tiếng Việt (tốt hơn cho đối tượng học viên). Fine-tuned code gọn hơn nhưng bỏ mất comments giải thích — đây là trường hợp **fine-tuned không cải thiện** so với base. Phản ánh bias của training dataset (English-only).

---

### Example 3 — List comprehension

**Prompt**: Sử dụng list comprehension để tạo danh sách các số chia hết cho 3 trong khoảng từ 1 đến 50.

**Base model**: `[3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]`

**Fine-tuned (r=16)**: `[3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]`

**Nhận xét**: **Kết quả hoàn toàn giống nhau** — cả hai đều ra kết quả trực tiếp thay vì viết code. Đây là trường hợp task đơn giản đến mức fine-tuning không tạo ra sự khác biệt. Base model đã đủ mạnh cho task này.

---

### Example 4 — Class SinhVien

**Prompt**: Viết một class `SinhVien` trong Python gồm các thuộc tính tên, tuổi và một phương thức để in thông tin.

**Base model**:
```python
class SinhVien:
    def __init__(self, ten, tuoi):
        self.ten = ten
        self.tuoi = tuoi

    def in_thong_tin(self):
        print(f'Tên: {self.ten}, Tuổi: {self.tuoi}')

s = SinhVien('Nguyen Van Be', 20)
s.in_thong_tin()
```

**Fine-tuned (r=16)**:
```python
class SinhVien:
    def __init__(self, ten, tuoi):
        self.ten = ten
        self.tuoi = tuoi

    def in_thongtin(self):
        print("Tên: " + self.ten)
        print("Tuổi: " + str(self.tuoi))

sv1 = SinhVien("Nguyen Van A", 18)
sv1.in_thongtin()
```

**Nhận xét**: Base model dùng f-string (Pythonic hơn, Python 3.6+). Fine-tuned dùng string concatenation (style cũ hơn). Đây là trường hợp **fine-tuned tệ hơn base** về code style. Có thể do training data có nhiều ví dụ dùng concatenation.

---

### Example 5 — Exception Handling

**Prompt**: Làm thế nào để bắt lỗi (Exception Handling) khi cố gắng mở một file không tồn tại?

**Base model**:
```python
try:
    file = open('file.txt', 'r')
except FileNotFoundError:
    print("File not found!")
else:
    file.read()
    file.close()
```

**Fine-tuned (r=16)**:
```python
try:
    file = open("example.txt", "r")
except FileNotFoundError:
    print("File not found.")

# This will print "File not found." when the file example.txt is not present.
```

**Nhận xét**: Base model đầy đủ hơn (dùng cả `else` clause để đóng file đúng cách). Fine-tuned đơn giản hơn nhưng **bỏ mất best practice** về resource management. Fine-tuned thêm comment giải thích (theo style training data) — nhưng về correctness, base tốt hơn.

**Tổng kết qualitative**: Trong 5 examples, fine-tuned model (r=16) **cải thiện nhẹ hoặc ngang bằng** base ở 2/5 cases (examples 1, 3), **tệ hơn** ở 3/5 cases (examples 2, 4, 5) về code quality. Điều này cho thấy với chỉ 500 samples, fine-tuning học được style của training dataset nhưng chưa đủ để consistently cải thiện reasoning hay best practices.

---

## 5. Conclusion về Rank Trade-off

Kết quả thực nghiệm cho thấy một bài học quan trọng: **rank cao hơn không đồng nghĩa với kết quả tốt hơn**, đặc biệt khi dataset nhỏ.

**ROI tốt nhất trên dataset này là r=8**. Với chỉ 500 samples (450 train), r=8 đạt perplexity tốt nhất (2.211), sử dụng ít VRAM nhất trong nhóm rank-experiment (5.74 GB), và hoàn thành training nhanh nhất (7.66 phút). Nguyên nhân cốt lõi là tỉ lệ trainable params so với số samples: r=64 có 97M params trên 450 examples (~216K params/sample), dẫn đến model memorize thay vì học pattern tổng quát.

**Diminishing returns xuất hiện rõ ràng từ r=16 lên r=64**. Từ r=8 lên r=16, perplexity tăng nhẹ (2.211 → 2.266, tức +2.5%), VRAM giảm nhẹ, thời gian tăng nhẹ — đây là trade-off có thể chấp nhận được nếu cần adapter linh hoạt hơn cho domain phức tạp. Nhưng từ r=16 lên r=64, perplexity tăng mạnh (2.266 → 2.481, +9.5%), VRAM tăng 46% (5.07 → 7.41 GB), trong khi training time không cải thiện. Đây là vùng diminishing returns rõ rệt — thêm params chỉ gây hại.

**Recommendation cho production**: Với dataset Python coding ~500 samples trên T4, **r=8 là lựa chọn tối ưu** về mọi chiều. Nếu dataset lớn hơn (>5,000 samples), r=16 hoặc r=32 sẽ thích hợp hơn vì model cần nhiều capacity để học diverse patterns. r=64 chỉ hợp lý khi có dataset rất lớn (>50k samples) và GPU mạnh (A100+) để tránh overfit. Quy tắc thực tế: **trainable params ≈ 10–50× số training samples** là vùng an toàn; vượt quá sẽ overfit.

---

## 6. What I Learned

- **Dataset size quyết định rank tối ưu, không phải ngược lại**: Trước lab này tôi nghĩ rank cao hơn luôn tốt hơn vì nhiều tham số hơn. Thực tế cho thấy r=64 tệ hơn r=8 trên 500 samples — bài học này thay đổi cách tôi sẽ chọn rank khi fine-tune trong tương lai: bao giờ cũng benchmark trên dataset thực tế thay vì dùng rank lớn mặc định.

- **QLoRA cho phép fine-tune trên hardware rất phổ thông**: Llama-3.2-3B với QLoRA 4-bit chỉ dùng ~5–7 GB VRAM, có nghĩa là T4 miễn phí trên Colab đủ để train một model hữu ích. Chi phí $0.14 để có 3 adapter checkpoints là minh chứng rõ ràng rằng fine-tuning không còn là đặc quyền của các công ty lớn.

- **Qualitative evaluation tiết lộ điều quantitative bỏ lỡ**: Perplexity của r=16 (2.266) trông ổn, nhưng qualitative comparison cho thấy fine-tuned model thực ra tệ hơn base về code style và best practices trong 3/5 cases. Perplexity chỉ đo xác suất token, không đo chất lượng thực sự của code. Kết hợp cả hai loại evaluation mới có bức tranh đầy đủ.
