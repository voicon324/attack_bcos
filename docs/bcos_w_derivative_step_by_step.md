# Giải thích từng bước về `W(x)` và đạo hàm theo `x` trong B-cos

Mục tiêu của file này là hiểu 4 ý:

1. `linear weight` là gì.
2. `B-cos scale` là gì.
3. `W(x)` là gì.
4. Đạo hàm `dW(x) / dx` đến từ đâu.

---

## 1. Bắt đầu với một neuron tuyến tính

Giả sử input là vector:

```math
x = [x_1, x_2, ..., x_d]
```

Neuron có weight:

```math
a = [a_1, a_2, ..., a_d]
```

Output tuyến tính bình thường là:

```math
z = a^T x
```

Tức là:

```math
z = a_1x_1 + a_2x_2 + ... + a_dx_d
```

Ở đây:

```text
linear weight = a
```

`a` là weight đã học của model. Khi model đã train xong, `a` được xem là cố định.

Với convolution, `a` chính là kernel, còn `x` là patch ảnh hoặc patch feature mà kernel đang nhìn vào.

---

## 2. B-cos thêm một hệ số scale

B-cos không chỉ dùng:

```math
a^T x
```

mà nhân thêm một hệ số phụ thuộc vào input:

```math
s(x)
```

Output B-cos có dạng:

```math
y = s(x) a^T x
```

Trong trường hợp thường gặp `B = 2`, B-cos scale là:

```math
s(x) = \frac{|a^T x|}{\|x\|}
```

Trong code có thêm số rất nhỏ `epsilon` để tránh chia cho 0:

```math
\|x\| = \sqrt{x^T x + \epsilon}
```

Vậy:

```math
\boxed{
s(x) = \frac{|a^T x|}{\sqrt{x^T x + \epsilon}}
}
```

Nói ngắn gọn:

```text
B-cos scale đo mức độ input x khớp hướng với weight a.
```

Nếu `x` càng cùng hướng với `a`, scale càng lớn.

Nếu `x` không khớp hướng với `a`, scale nhỏ hơn.

---

## 3. `W(x)` với một B-cos unit là gì?

Ta có output:

```math
y = s(x) a^T x
```

Vì `s(x)` là một số, ta có thể viết lại:

```math
y = (s(x)a)^T x
```

Vậy weight hiệu dụng tại input `x` là:

```math
W(x) = s(x)a
```

Thay công thức `s(x)` vào:

```math
\boxed{
W(x) =
\frac{|a^T x|}{\sqrt{x^T x + \epsilon}} a
}
```

Đây là ý quan trọng nhất:

```text
W(x) = B-cos scale * linear weight
```

hay:

```text
W(x) = s(x) * a
```

---

## 4. Đạo hàm của B-cos scale

Ta cần tính đạo hàm của:

```math
s(x) = \frac{|a^T x|}{\sqrt{x^T x + \epsilon}}
```

Đặt:

```math
r = a^T x
```

và:

```math
n = \sqrt{x^T x + \epsilon}
```

Khi đó:

```math
s(x) = \frac{|r|}{n}
```

Đạo hàm theo `x` là:

```math
\boxed{
\nabla_x s(x)
=
\frac{\operatorname{sign}(a^T x)a}{\sqrt{x^T x + \epsilon}}
-
\frac{|a^T x|x}{(x^T x + \epsilon)^{3/2}}
}
```

Nếu viết theo từng phần tử `x_j`:

```math
\boxed{
\frac{\partial s(x)}{\partial x_j}
=
\frac{\operatorname{sign}(a^T x)a_j}{\sqrt{x^T x + \epsilon}}
-
\frac{|a^T x|x_j}{(x^T x + \epsilon)^{3/2}}
}
```

Trong công thức này:

```text
a_j là weight thứ j, cố định.
x_j là input thứ j, thay đổi theo ảnh.
a^T x thay đổi theo input.
x^T x thay đổi theo input.
```

---

## 5. Đạo hàm của `W(x)` với một B-cos unit

Ta đã có:

```math
W(x) = s(x)a
```

Vì `a` cố định, đạo hàm của `W(x)` chỉ đến từ `s(x)`:

```math
\frac{dW(x)}{dx}
=
a \frac{ds(x)}{dx}
```

Viết theo từng phần tử:

```math
\boxed{
\frac{\partial W_i(x)}{\partial x_j}
=
a_i
\left[
\frac{\operatorname{sign}(a^T x)a_j}{\sqrt{x^T x + \epsilon}}
-
\frac{|a^T x|x_j}{(x^T x + \epsilon)^{3/2}}
\right]
}
```

Đọc bằng lời:

```text
Đạo hàm của W(x)
= linear weight
  nhân với
  đạo hàm của B-cos scale.
```

---

## 6. Với nhiều layer thì `W(x)` lấy thế nào?

Với nhiều layer, mỗi layer có:

```text
linear weight của layer đó
```

và:

```text
B-cos scale của layer đó
```

Ký hiệu:

```text
A_l = linear weight của layer l
D_l(x) = ma trận chứa các B-cos scale của layer l
```

Với mạng 3 layer:

```math
h_1 = D_1(x) A_1 x
```

```math
h_2 = D_2(x) A_2 h_1
```

```math
f_c = D_3(x) A_3 h_2
```

Nếu cần explanation weight cho class `c`, ta đi ngược từ class `c` về input:

```math
W_c(x)
=
A_1^T D_1(x)^T
A_2^T D_2(x)^T
A_3^T D_3(x)^T
e_c
```

Với `L` layer:

```math
\boxed{
W_c(x)
=
A_1^T D_1(x)^T
A_2^T D_2(x)^T
\cdots
A_L^T D_L(x)^T
e_c
}
```

Trong đó:

```text
e_c là vector chọn class c.
```

Nói dễ hiểu:

```text
W_c(x) là effective weight từ input đến class c.
Nó được tạo bằng cách nhân ngược các weight và scale từ output về input.
```

---

## 7. Đạo hàm `W(x)` với nhiều layer

Ta có:

```math
W_c(x)
=
A_1^T D_1(x)^T
A_2^T D_2(x)^T
\cdots
A_L^T D_L(x)^T
e_c
```

Các `A_l` là weight đã học, nên cố định.

Các `D_l(x)` là B-cos scale, nên phụ thuộc vào input `x`.

Vì vậy, đạo hàm `dW/dx` đến từ việc các `D_l(x)` thay đổi khi `x` thay đổi.

Công thức:

```math
\boxed{
\frac{\partial W_c(x)}{\partial x_j}
=
\sum_{k=1}^{L}
A_1^T D_1(x)^T
\cdots
A_k^T
\frac{\partial D_k(x)^T}{\partial x_j}
\cdots
A_L^T D_L(x)^T
e_c
}
```

Nói bằng lời:

```text
Đạo hàm của W(x)
= tổng các ảnh hưởng do từng B-cos scale ở từng layer thay đổi.
```

---

## 8. Liên hệ với code

Trong code, `W(x)` là biến `weights`:

```python
weights = torch.autograd.grad(linear_score, x_linear, create_graph=True)[0]
```

Về toán học:

```math
W_c(x) = \nabla_x f_c^{frozen}(x)
```

`frozen` nghĩa là:

```text
Model lấy các B-cos scale tại input x,
rồi xem các scale đó như hệ số của một mạng tuyến tính.
```

Khi `create_graph=True`, PyTorch giữ lại graph của `W(x)`.

Vì vậy nếu loss phụ thuộc vào `W(x)`, PyTorch có thể tính:

```math
\frac{d}{dx} Loss(W(x))
```

theo chain rule:

```math
\frac{d Loss}{dx}
=
\frac{d Loss}{d W}
\frac{d W}{d x}
```

---

## 9. Tóm tắt ngắn nhất

Với một B-cos unit:

```math
W(x) = s(x)a
```

trong đó:

```math
s(x) = \frac{|a^T x|}{\sqrt{x^T x + \epsilon}}
```

Đạo hàm của scale:

```math
\nabla_x s(x)
=
\frac{\operatorname{sign}(a^T x)a}{\sqrt{x^T x + \epsilon}}
-
\frac{|a^T x|x}{(x^T x + \epsilon)^{3/2}}
```

Đạo hàm của `W(x)`:

```math
\frac{dW}{dx}
=
a \frac{ds}{dx}
```

Với nhiều layer:

```math
W_c(x)
=
A_1^T D_1(x)^T
A_2^T D_2(x)^T
\cdots
A_L^T D_L(x)^T
e_c
```

Và:

```text
dW/dx là tổng các đạo hàm do từng B-cos scale D_l(x) gây ra.
```

