// Thanks to: http://www.mm.media.kyoto-u.ac.jp/education/le4dip/
#define NDARRAY_HPP_ENABLE_CBLAS
#include <ndarray.hpp>
using namespace ndarray_hpp;

//
#include <algorithm>
#include <any>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <tuple>

///////////////////////
//// Random generator
//////////////////////

class Random {
private:
    static std::mt19937 engine_;

public:
    static auto& engine()
    {
        return engine_;
    }

    template <class Dist>
    static auto rand(Dist dist)
    {
        return dist(engine_);
    }
};
// std::mt19937 Random::engine_{std::random_device{}()};
std::mt19937 Random::engine_{0};

///////////////////////
//// N-by-M matrix
//////////////////////
using DynamicMatrix = std::any;

template <class Float, size_t... Shape>
ndarray<Float, Shape...> mat_cast(const DynamicMatrix& src)
{
    return std::any_cast<ndarray<Float, Shape...>>(src);
}

/////////////////////
//// Utilities
/////////////////////
template <size_t N = 0, class Func, class... Types>
void visit_tuple(std::tuple<Types...>& tup, Func func)
{
    if constexpr (N < sizeof...(Types)) {
        func(std::get<N>(tup));
        visit_tuple<N + 1>(tup, func);
    }
}

// Thanks to: https://qiita.com/IgnorantCoder/items/a6cebba4de6cb5901335
// Thanks to: https://qiita.com/tyanmahou/items/c2a7c389e666d1b1bdea
// Thanks to: https://cpprefjp.github.io/reference/type_traits/void_t.html
template <class, template <class...> class, class...>
struct is_detected_impl : std::false_type {
};
template <template <class...> class Op, class... Args>
struct is_detected_impl<std::void_t<Op<Args...>>, Op, Args...>
    : std::true_type {
};
template <template <class...> class Op, class... Args>
using is_detected = is_detected_impl<void, Op, Args...>;

/////////////////////
//// NN component
/////////////////////
template <class Float, size_t OutSize, size_t... InSize>
struct Linear {
    Matrix<Float, (InSize * ...), OutSize> W, dW;
    Vector<Float, OutSize> b, db;
    DynamicMatrix x;

    template <size_t BatchSize>
    Matrix<Float, BatchSize, OutSize> forward(
        const ndarray<Float, BatchSize, InSize...>& src)
    {
        auto x = src.template reshape<BatchSize, (InSize * ...)>();
        this->x = x;
        return dot(x, W) + b;
    }

    template <size_t BatchSize>
    ndarray<Float, BatchSize, InSize...> backward(
        const Matrix<Float, BatchSize, OutSize>& src)
    {
        auto x = mat_cast<Float, BatchSize, (InSize * ...)>(this->x);

        auto dx = dot(src, transpose(W));
        dW = dot(transpose(x), src);
        db = sum<0>(src);

        return dx.template reshape<BatchSize, InSize...>();
    }

    template <class Factory>
    auto optimizers(Factory factory)
    {
        return std::make_tuple(factory(W, dW), factory(b, db));
    }
};

template <class Float, size_t InSize, size_t OutSize>
using Linear2 = Linear<Float, OutSize, InSize>;

template <class Float, size_t... InSize>
struct BatchNormalization {
    static constexpr size_t InSizeAll = (InSize * ...);
    Vector<Float, InSizeAll> gamma = Vector<Float, InSizeAll>::ones(),
                             beta = Vector<Float, InSizeAll>::zeros(),
                             dgamma = Vector<Float, InSizeAll>::zeros(),
                             dbeta = Vector<Float, InSizeAll>::zeros();

    Vector<Float, InSizeAll> E_x = Vector<Float, InSizeAll>::zeros(),
                             V_x = Vector<Float, InSizeAll>::zeros();
    size_t numOfForward = 0;

    DynamicMatrix x, mu, v, xhat;

    bool is_training = true;

    template <size_t BatchSize>
    ndarray<Float, BatchSize, InSize...> forward(
        ndarray<Float, BatchSize, InSize...> src0)
    {
        const static Float eps = 1e-7;

        auto src = src0.template reshape<BatchSize, InSizeAll>();

        if (is_training) {
            auto mu = sum<0>(src) / BatchSize;
            auto v = sum<0>((src - mu) * (src - mu)) / BatchSize;
            auto xhat = (src - mu) / sqrt(v + eps);
            auto y = xhat * gamma + beta;

            Float n = numOfForward;
            E_x = E_x * (n / (n + 1)) + mu / (n + 1);
            V_x = V_x * (n / (n + 1)) + v / (n + 1);
            numOfForward++;

            this->x = src;
            this->mu = mu;
            this->v = v;
            this->xhat = xhat;

            return y.template reshape<BatchSize, InSize...>();
        }
        else {
            auto tmp = (src - E_x) * gamma / sqrt(V_x + eps) + beta;
            return tmp.template reshape<BatchSize, InSize...>();
        }
    }

    template <size_t BatchSize>
    ndarray<Float, BatchSize, InSize...> backward(
        ndarray<Float, BatchSize, InSize...> dE_dy0)
    {
        const static Float eps = 1e-7;
        auto x = mat_cast<Float, BatchSize, InSizeAll>(this->x);
        auto mu = mat_cast<Float, InSizeAll>(this->mu);
        auto v = mat_cast<Float, InSizeAll>(this->v);
        auto xhat = mat_cast<Float, BatchSize, InSizeAll>(this->xhat);

        auto dE_dy = dE_dy0.template reshape<BatchSize, InSizeAll>();
        auto dE_dxhat = dE_dy * gamma;
        auto dE_dv = -sum<0>(dE_dxhat * (x - mu) * pow(v + eps, -3.f / 2)) / 2;
        auto dE_dmu = -sum<0>(dE_dxhat / sqrt(v + eps)) -
                      dE_dv * sum<0>(x - mu) * (2. / BatchSize);
        auto dE_dx = dE_dxhat / sqrt(v + eps) +
                     (x - mu) * dE_dv * (2. / BatchSize) + dE_dmu / BatchSize;
        dgamma = sum<0>(dE_dy * xhat);
        dbeta = sum<0>(dE_dy);

        return dE_dx.template reshape<BatchSize, InSize...>();
    }

    template <class Factory>
    auto optimizers(Factory factory)
    {
        return std::make_tuple(factory(gamma, dgamma), factory(beta, dbeta));
    }
};

template <class Float, size_t InSize>
struct SoftmaxCrossEntropy {
    DynamicMatrix y, t;

    template <size_t BatchSize>
    Float forward(Matrix<Float, BatchSize, InSize> src,
                  const Matrix<Float, BatchSize, InSize>& t)
    {
        // process softmax
        for (size_t i = 0; i < BatchSize; i++) {
            // c = max src[i]
            Float c = src(i, 0);
            for (size_t j = 0; j < InSize; j++)
                c = std::max(c, src(i, j));

            // src[i] = exp(src[i] - c)
            // s = sum src[i]
            Float s = 0;
            for (size_t j = 0; j < InSize; j++) {
                src(i, j) = std::exp(src(i, j) - c);
                s += src(i, j);
            }

            // src[i] /= s
            for (size_t j = 0; j < InSize; j++)
                src(i, j) /= s;
        }
        this->y = src;
        this->t = t;

        // process cross-entropy
        Float loss_sum = 0;
        const static Float delta = 1e-7;
        for (size_t i = 0; i < BatchSize; i++)
            for (size_t j = 0; j < InSize; j++)
                loss_sum += t(i, j) * std::log(src(i, j) + delta);

        return -loss_sum / BatchSize;
    }

    template <size_t BatchSize>
    Matrix<Float, BatchSize, InSize> backward()
    {
        auto y = mat_cast<Float, BatchSize, InSize>(this->y);
        auto t = mat_cast<Float, BatchSize, InSize>(this->t);
        Matrix<Float, BatchSize, InSize> dx;
        for (size_t i = 0; i < BatchSize; i++)
            for (size_t j = 0; j < InSize; j++)
                dx(i, j) = (y(i, j) - t(i, j)) / BatchSize;
        return dx;
    }
};

template <class Float, size_t InSize>
struct Sigmoid {
    DynamicMatrix y;

    template <size_t BatchSize>
    Matrix<Float, BatchSize, InSize> forward(
        Matrix<Float, BatchSize, InSize> src)
    {
        src.each([&](auto i, auto j) {
            src(i, j) = 1. / (1 + std::exp(-src(i, j)));
        });

        y = src;

        return src;
    }

    template <size_t BatchSize>
    Matrix<Float, BatchSize, InSize> backward(
        Matrix<Float, BatchSize, InSize> src)
    {
        auto y = mat_cast<BatchSize, InSize>(this->y);
        src.each([&](auto i, auto j) { src(i, j) *= y(i, j) * (1 - y(i, j)); });
        return src;
    }
};

template <class Float, size_t... InSize>
struct ReLU {
    DynamicMatrix y;

    template <size_t BatchSize>
    ndarray<Float, BatchSize, InSize...> forward(
        ndarray<Float, BatchSize, InSize...> src)
    {
        src.each([&](auto&&... indices) {
            if (src(indices...) <= 0)
                src(indices...) = 0;
        });

        y = src;

        return src;
    }

    template <size_t BatchSize>
    ndarray<Float, BatchSize, InSize...> backward(
        ndarray<Float, BatchSize, InSize...> src)
    {
        auto y = std::any_cast<ndarray<Float, BatchSize, InSize...>>(this->y);
        src.each([&](auto&&... indices) {
            if (y(indices...) <= 0)
                src(indices...) = 0;
        });
        return src;
    }
};

template <class Float, size_t InSize>
struct Dropout {
    Float ratio = 0.5;
    bool is_training = true;
    DynamicMatrix y;

    template <size_t BatchSize>
    Matrix<Float, BatchSize, InSize> forward(
        Matrix<Float, BatchSize, InSize> src)
    {
        static const std::uniform_real_distribution<> dist(0.0, 1.0);

        if (is_training) {
            src.mask_v([&](auto) { return Random::rand(dist) > ratio; });
            y = src;
        }
        else {
            src.map_v([&](auto val) { return val * (1.0 - ratio); });
        }

        return src;
    }

    template <size_t BatchSize>
    Matrix<Float, BatchSize, InSize> backward(
        Matrix<Float, BatchSize, InSize> src)
    {
        auto y = mat_cast<Float, BatchSize, InSize>(this->y);
        src.mask_i([&](auto i, auto j) { return y(i, j) > 0; });
        return src;
    }
};

namespace detail {
template <size_t Pad, class Float, size_t N, size_t C, size_t H, size_t W>
struct padding_view4 {
    const ndarray<Float, N, C, H, W>& src;
    Float fill;

    Float operator()(size_t i, size_t j, size_t k, size_t l) const
    {
        if (k < Pad || l < Pad || k >= H + Pad || l >= W + Pad)
            return fill;
        return src(i, j, k - Pad, l - Pad);
    }
};

template <size_t Pad, class Float, size_t N, size_t C, size_t H, size_t W>
struct padding_view4_w {
    ndarray<Float, N, C, H, W>& src;
    Float fill = 0;

    Float& operator()(size_t i, size_t j, size_t k, size_t l)
    {
        if (k < Pad || l < Pad || k >= H + Pad || l >= W + Pad)
            return fill;
        return src(i, j, k - Pad, l - Pad);
    }
};

constexpr size_t get_out_size(size_t S, size_t F, size_t Stride, size_t Pad)
{
    return (S + 2 * Pad - F) / Stride + 1;
}

}  // namespace detail

template <size_t FH, size_t FW, size_t Stride, size_t Pad, size_t N, size_t C,
          size_t H, size_t W, class Float>
auto im2col(const ndarray<Float, N, C, H, W>& src)
{
    constexpr auto out_h = ::detail::get_out_size(H, FH, Stride, Pad);
    constexpr auto out_w = ::detail::get_out_size(W, FW, Stride, Pad);

    auto img = ::detail::padding_view4<Pad, Float, N, C, H, W>{src, 0};
    auto col = ndarray<Float, N * out_h * out_w, C * FH * FW>::zeros();

    for (size_t i = 0; i < N * C * FH * FW * out_h * out_w; i++) {
        auto out_x = i % out_w;
        auto out_y = (i / out_w) % out_h;
        auto fx = (i / out_w / out_h) % FW;
        auto fy = (i / out_w / out_h / FW) % FH;
        auto ci = (i / out_w / out_h / FW / FH) % C;
        auto ni = (i / out_w / out_h / FW / FH / C) % N;

        col((ni * out_h + out_y) * out_w + out_x, (ci * FH + fy) * FW + fx) =
            img(ni, ci, out_y * Stride + fy, out_x * Stride + fx);
    }

    return col;
}

template <size_t FH, size_t FW, size_t Stride, size_t Pad, size_t N, size_t C,
          size_t H, size_t W, class Float,
          size_t out_h = ::detail::get_out_size(H, FH, Stride, Pad),
          size_t out_w = ::detail::get_out_size(H, FH, Stride, Pad)>
auto col2im(const ndarray<Float, N * out_h * out_w, C * FH * FW>& col)
{
    auto ret = ndarray<Float, N, C, H, W>::zeros();
    auto img = ::detail::padding_view4_w<Pad, Float, N, C, H, W>{ret};

    for (size_t i = 0; i < N * C * FH * FW * out_h * out_w; i++) {
        auto out_x = i % out_w;
        auto out_y = (i / out_w) % out_h;
        auto fx = (i / out_w / out_h) % FW;
        auto fy = (i / out_w / out_h / FW) % FH;
        auto ci = (i / out_w / out_h / FW / FH) % C;
        auto ni = (i / out_w / out_h / FW / FH / C) % N;

        img(ni, ci, out_y * Stride + fy, out_x * Stride + fx) +=
            col((ni * out_h + out_y) * out_w + out_x, (ci * FH + fy) * FW + fx);
    }

    return ret;
}

template <class Float, size_t FN, size_t FH, size_t FW, size_t Stride,
          size_t Pad, size_t C, size_t H, size_t W,
          size_t OUT_H = ::detail::get_out_size(H, FH, Stride, Pad),
          size_t OUT_W = ::detail::get_out_size(W, FW, Stride, Pad)>
struct Convolution2D {
    ndarray<Float, FN, C, FH, FW> We, dWe;
    Vector<Float, FN> b, db;

    DynamicMatrix x, col, col_W;

    template <size_t BatchSize>
    ndarray<Float, BatchSize, FN, OUT_H, OUT_W> forward(
        const ndarray<Float, BatchSize, C, H, W>& src)
    {
        auto col = im2col<FH, FW, Stride, Pad>(src);
        auto col_W = transpose(We.template reshape<FN, C * FH * FW>());
        auto out = (dot(col, col_W) + b)
                       .template reshape<BatchSize, OUT_H, OUT_W, FN>();

        // ret = out.transpose(0, 3, 1, 2)
        ndarray<Float, BatchSize, FN, OUT_H, OUT_W> ret;
        static_assert(ret.size() == out.size());
        out.each([&](auto i, auto j, auto k, auto l) {
            ret(i, l, j, k) = out(i, j, k, l);
        });

        this->x = src;
        this->col = col;
        this->col_W = col_W;

        return ret;
    }

    template <size_t BatchSize>
    auto backward(const ndarray<Float, BatchSize, FN, OUT_H, OUT_W>& src)
    {
        auto x = std::any_cast<ndarray<Float, BatchSize, C, H, W>>(this->x);
        auto col_T = transpose(
            mat_cast<Float, BatchSize * OUT_H * OUT_W, C * FH * FW>(this->col));
        auto col_W_T = transpose(mat_cast<Float, C * FH * FW, FN>(this->col_W));

        // dout0 = src.transpose(0, 2, 3, 1)
        ndarray<Float, BatchSize, OUT_H, OUT_W, FN> dout0;
        static_assert(std::decay_t<decltype(dout0)>::size() ==
                      std::decay_t<decltype(src)>::size());
        src.each([&](auto i, auto j, auto k, auto l) {
            dout0(i, k, l, j) = src(i, j, k, l);
        });

        auto dout = dout0.template reshape<BatchSize * OUT_H * OUT_W, FN>();
        auto dcol = dot(dout, col_W_T);
        auto dx = col2im<FH, FW, Stride, Pad, BatchSize, C, H, W>(dcol);

        db = sum<0>(dout);
        dWe = transpose(dot(col_T, dout)).template reshape<FN, C, FH, FW>();

        return dx;
    }

    template <class Factory>
    auto optimizers(Factory factory)
    {
        return std::make_tuple(factory(We, dWe), factory(b, db));
    }
};

template <class Float, size_t PH, size_t PW, size_t Stride, size_t Pad,
          size_t C, size_t H, size_t W,
          size_t OUT_H = ::detail::get_out_size(H, PH, Stride, 0),
          size_t OUT_W = ::detail::get_out_size(W, PW, Stride, 0)>
struct MaxPooling2D {
    static_assert(Pad == 0);  // FIXME: relax this condition.

    DynamicMatrix x, arg_max;

    template <size_t BatchSize>
    ndarray<Float, BatchSize, C, OUT_H, OUT_W> forward(
        const ndarray<Float, BatchSize, C, H, W>& src)
    {
        ndarray<Float, BatchSize, C, OUT_H, OUT_W> max;
        ndarray<size_t, BatchSize, C, OUT_H, OUT_W> arg_max;

        for (size_t i = 0; i < BatchSize * C * OUT_H * OUT_W; i++) {
            auto x = i % OUT_W;
            auto y = (i / OUT_W) % OUT_H;
            auto ch = (i / OUT_W / OUT_H) % C;
            auto n = (i / OUT_W / OUT_H / C) % BatchSize;

            // Finx max value in the filter.
            Float max_val = src(n, ch, 0, 0);
            size_t arg_max_val = 0;
            for (size_t fy = 0; fy < PH; fy++) {
                for (size_t fx = 0; fx < PW; fx++) {
                    Float val = src(n, ch, Stride * y + fy, Stride * x + fx);
                    if (val <= max_val)
                        continue;
                    max_val = val;
                    arg_max_val = fy * PW + fx;
                }
            }

            max(n, ch, y, x) = max_val;
            arg_max(n, ch, y, x) = arg_max_val;
        }

        this->arg_max = arg_max;

        return max;
    }

    template <size_t BatchSize>
    ndarray<Float, BatchSize, C, H, W> backward(
        const ndarray<Float, BatchSize, C, OUT_H, OUT_W>& src)
    {
        auto arg_max =
            std::any_cast<ndarray<size_t, BatchSize, C, OUT_H, OUT_W>>(
                this->arg_max);

        auto ret = ndarray<Float, BatchSize, C, H, W>::zeros();
        for (size_t i = 0; i < BatchSize * C * OUT_H * OUT_W; i++) {
            auto x = i % OUT_W;
            auto y = (i / OUT_W) % OUT_H;
            auto ch = (i / OUT_W / OUT_H) % C;
            auto n = (i / OUT_W / OUT_H / C) % BatchSize;

            size_t arg = arg_max(n, ch, y, x);
            ret(n, ch, Stride * y + arg / PW, Stride * x + arg % PW) =
                src(n, ch, y, x);
        }

        return ret;
    }
};

/////////////////////
//// Neural Networks
/////////////////////

// Check if T has is_training
template <class T>
using has_is_training_impl = decltype(std::declval<T>().is_training);
template <class T>
using has_is_training = is_detected<has_is_training_impl, T>;

// Check if T has optimizers(Factory)
template <class T, class Factory>
using has_optimizers_impl =
    decltype(std::declval<T>().optimizers(std::declval<Factory>()));
template <class T, class Factory>
using has_optimizers = is_detected<has_optimizers_impl, T, Factory>;

template <class>
struct is_linear : std::false_type {
};
template <class Float, size_t OutSize, size_t... InSize>
struct is_linear<Linear<Float, OutSize, InSize...>> : std::true_type {
};
template <class>
struct is_convolution2d : std::false_type {
};
template <class Float, size_t... Shape>
struct is_convolution2d<Convolution2D<Float, Shape...>> : std::true_type {
};

template <class Float, size_t OutSize, size_t... InSize>
void initLinear(Linear<Float, OutSize, InSize...>& l)
{
    constexpr auto AllInSize = (InSize * ...);
    l.W = Matrix<Float, AllInSize, OutSize>::randn(Random::engine(), 0,
                                                   1. / AllInSize);
    l.b = Vector<Float, OutSize>::randn(Random::engine(), 0, 1. / AllInSize);
}

template <class Float, size_t FN, size_t FH, size_t FW, size_t Stride,
          size_t Pad, size_t C, size_t H, size_t W>
void initConv(Convolution2D<Float, FN, FH, FW, Stride, Pad, C, H, W>& l)
{
    l.We = ndarray<Float, FN, C, FH, FW>::randn(Random::engine(), 0,
                                                1. / (C * FH * FW));
    l.b = ndarray<Float, FN>::randn(Random::engine(), 0, 1. / (C * FH * FW));
}

template <class Float, class NetworkStack>
struct MLP {
    using type = Float;
    NetworkStack stack;
    constexpr static size_t depth = std::tuple_size<NetworkStack>::value;

    MLP()
    {
        each_linear([](auto&& linear) { initLinear(linear); });
        each_convolution2d([](auto&& conv) { initConv(conv); });
    }

    template <class Input>
    auto predict(Input&& src)
    {
        turn_training(false);
        return forward_detail<0, depth - 1>(std::forward<Input>(src));
    }

    template <class Input, class Answer>
    auto forward(Input&& src, Answer&& t)
    {
        turn_training(true);
        return std::get<depth - 1>(stack).forward(
            forward_detail<0, depth - 1>(std::forward<Input>(src)),
            std::forward<Answer>(t));
    }

    template <size_t BatchSize>
    auto backward()
    {
        return backward_detail<BatchSize>();
    }

    template <size_t BatchSize, size_t... InSize, size_t... OutSize>
    Float step(const ndarray<Float, BatchSize, InSize...>& src,
               const ndarray<Float, BatchSize, OutSize...>& t)
    {
        Float loss = forward(src, t);
        backward<BatchSize>();
        return loss;
    }

    template <class Factory>
    auto optimizers(Factory factory)
    {
        return optimizers_detail(factory);
    }

private:
    template <size_t N = 0>
    void turn_training(bool on)
    {
        if constexpr (N < depth) {
            auto& layer = std::get<N>(stack);
            if constexpr (has_is_training<decltype(
                              std::get<N>(stack))>::value) {
                layer.is_training = on;
            }
            turn_training<N + 1>(on);
        }
    }

    template <class Func>
    void each_linear(Func func)
    {
        visit_tuple(stack, [&](auto&& layer) {
            using T = std::decay_t<decltype(layer)>;
            if constexpr (is_linear<T>()) {
                func(layer);
            }
        });
    }

    template <class Func>
    void each_convolution2d(Func func)
    {
        visit_tuple(stack, [&](auto&& layer) {
            using T = std::decay_t<decltype(layer)>;
            if constexpr (is_convolution2d<T>()) {
                func(layer);
            }
        });
    }

    template <size_t N, size_t Limit, class Input>
    auto forward_detail(Input input)
    {
        static_assert(Limit <= depth);

        if constexpr (N < Limit) {
            return forward_detail<N + 1, Limit>(
                std::get<N>(stack).forward(input));
        }
        else {
            return input;
        }
    }

    template <size_t N, class Input>
    auto backward_detail(Input&& src)
    {
        static_assert(0 <= N && N < depth - 1);

        if constexpr (N == 0) {
            return std::get<0>(stack).backward(std::forward<Input>(src));
        }
        else {
            return backward_detail<N - 1>(
                std::get<N>(stack).backward(std::forward<Input>(src)));
        }
    }

    template <size_t BatchSize>
    auto backward_detail()
    {
        static_assert(2 <= depth);
        return backward_detail<depth - 2>(
            std::get<depth - 1>(stack).template backward<BatchSize>());
    }

    template <size_t N = 0, class Factory>
    auto optimizers_detail(Factory factory)
    {
        if constexpr (N < depth) {
            using T = std::decay_t<decltype(std::get<N>(stack))>;
            if constexpr (has_optimizers<T, Factory>::value) {
                auto& layer = std::get<N>(stack);
                return std::tuple_cat(layer.optimizers(factory),
                                      optimizers_detail<N + 1>(factory));
            }
            else {
                return optimizers_detail<N + 1>(factory);
            }
        }
        else {
            return std::make_tuple();
        }
    }
};

/////////////////////
//// Helper functions to define NN like define-by-run.
/////////////////////
template <template <class, size_t...> class Layer, class T, size_t... Shape>
T get_type_impl(Layer<T, Shape...>);
template <class T>
using get_type = decltype(get_type_impl(std::declval<T>()));

template <class T, class Tuple>
decltype(std::tuple_cat(std::declval<std::tuple<T>>(), std::declval<Tuple>()))
tuple_cons();

template <class Float, size_t BatchSize, size_t... OutSize>
std::index_sequence<OutSize...> get_output_shape_impl(
    ndarray<Float, BatchSize, OutSize...>);

template <class R, class... Types>
R rettype(R (*)(Types...));
template <class C, class R, class... Types>
R rettype(R (C::*)(Types...));

template <class T, size_t BatchSize = 100>
using get_output_shape =
    decltype(get_output_shape_impl(rettype(&T::template forward<BatchSize>)));

template <class Float, size_t OutSize, size_t... PrevOutSize>
Linear<Float, OutSize, PrevOutSize...> linear_impl(
    std::index_sequence<PrevOutSize...>);

template <size_t OutSize, class Prev, class... Types>
auto linear(std::tuple<Prev, Types...>)
{
    return decltype(tuple_cons<decltype(linear_impl<get_type<Prev>, OutSize>(
                                   get_output_shape<Prev>())),
                               std::tuple<Prev, Types...>>()){};
}

template <class Float, size_t... PrevOutSize>
ReLU<Float, PrevOutSize...> relu_impl(std::index_sequence<PrevOutSize...>);
template <class Prev, class... Types>
auto relu(std::tuple<Prev, Types...>)
{
    return decltype(tuple_cons<decltype(relu_impl<get_type<Prev>>(
                                   get_output_shape<Prev>())),
                               std::tuple<Prev, Types...>>()){};
}

template <class Float, size_t... PrevOutSize>
Dropout<Float, PrevOutSize...> dropout_impl(
    std::index_sequence<PrevOutSize...>);
template <class Prev, class... Types>
auto dropout(std::tuple<Prev, Types...>)
{
    return decltype(tuple_cons<decltype(dropout_impl<get_type<Prev>>(
                                   get_output_shape<Prev>())),
                               std::tuple<Prev, Types...>>()){};
}

template <class Float, size_t... PrevOutSize>
BatchNormalization<Float, PrevOutSize...> batch_normalization_impl(
    std::index_sequence<PrevOutSize...>);
template <class Prev, class... Types>
auto batch_normalization(std::tuple<Prev, Types...>)
{
    return decltype(
        tuple_cons<decltype(batch_normalization_impl<get_type<Prev>>(
                       get_output_shape<Prev>())),
                   std::tuple<Prev, Types...>>()){};
}

template <class Float, size_t PrevOutSize>
SoftmaxCrossEntropy<Float, PrevOutSize> softmax_cross_entropy_impl(
    std::index_sequence<PrevOutSize>);
template <class Prev, class... Types>
auto softmax_cross_entropy(std::tuple<Prev, Types...>)
{
    return decltype(
        tuple_cons<decltype(softmax_cross_entropy_impl<get_type<Prev>>(
                       get_output_shape<Prev>())),
                   std::tuple<Prev, Types...>>()){};
}

template <class Float, size_t FN, size_t FW, size_t FH, size_t Stride,
          size_t Pad, size_t C, size_t W, size_t H>
Convolution2D<Float, FN, FH, FW, Stride, Pad, C, H, W> convolution_2d_impl(
    std::index_sequence<C, H, W>);
template <size_t FN, size_t FW, size_t FH, size_t Stride, size_t Pad,
          class Prev, class... Types>
auto convolution_2d(std::tuple<Prev, Types...>)
{
    return decltype(
        tuple_cons<decltype(
                       convolution_2d_impl<get_type<Prev>, FN, FW, FH, Stride,
                                           Pad>(get_output_shape<Prev>())),
                   std::tuple<Prev, Types...>>()){};
}

template <class Float, size_t PW, size_t PH, size_t Stride, size_t Pad,
          size_t C, size_t W, size_t H>
MaxPooling2D<Float, PH, PW, Stride, Pad, C, H, W> max_pooling_2d_impl(
    std::index_sequence<C, H, W>);
template <size_t PW, size_t PH, size_t Stride, size_t Pad, class Prev,
          class... Types>
auto max_pooling_2d(std::tuple<Prev, Types...>)
{
    return decltype(
        tuple_cons<decltype(max_pooling_2d_impl<get_type<Prev>, PW, PH, Stride,
                                                Pad>(get_output_shape<Prev>())),
                   std::tuple<Prev, Types...>>()){};
}

template <class Float, size_t... OutSize>
struct Sentinel {
    template <size_t BatchSize>
    ndarray<Float, BatchSize, OutSize...> forward();
};

template <class T, class... Types>
auto define_nn_impl()
{
    if constexpr (sizeof...(Types) == 0) {
        return std::make_tuple();
    }
    else {
        return std::tuple_cat(define_nn_impl<Types...>(), std::tuple<T>());
    }
}

template <class... Types>
decltype(define_nn_impl<Types...>()) define_nn_impl(std::tuple<Types...>);

template <class Schema>
auto define_nn(Schema schema)
{
    using NetworkStack = decltype(define_nn_impl(rettype(schema)));
    using Float = get_type<decltype(std::get<0>(std::declval<NetworkStack>()))>;
    return MLP<Float, NetworkStack>{};
}

template <class Float, size_t... Shape>
using InputImage = std::tuple<Sentinel<Float, Shape...>>;

/////////////////////
//// Optimizers
/////////////////////
template <class Float, class Mat>
struct SGD {
    Float lr;
    Mat &W, &dW;

    SGD(Mat& W, Mat& dW, Float lr) : lr(lr), W(W), dW(dW)
    {
    }

    void operator()()
    {
        W = W + dW * -lr;
    }
};

template <class Float, class Mat>
struct MomentumSGD {
    Float eta, alpha;
    Mat &W, &dW, d;

    MomentumSGD(Mat& W, Mat& dW, Float eta, Float alpha)
        : eta(eta), alpha(alpha), W(W), dW(dW), d(Mat::zeros())
    {
    }

    void operator()()
    {
        d = d * alpha + dW * -eta;
        W = W + d;
    }
};

template <class Float, class Mat>
struct Adam {
    Float alpha, beta1, beta2, beta1_t, beta2_t, eps;
    Mat &W, &dW;
    Mat m, v;

    Adam(Mat& W, Mat& dW, Float a, Float b1, Float b2, Float e)
        : alpha(a),
          beta1(b1),
          beta2(b2),
          beta1_t(b1),
          beta2_t(b2),
          eps(e),
          W(W),
          dW(dW)
    {
    }

    void operator()()
    {
        m = m * beta1 + dW * (1 - beta1);
        v = v * beta2 + dW * dW * (1 - beta2);
        auto m_hat = m / (1 - beta1_t);
        auto v_hat = v / (1 - beta2_t);
        W = W - (m_hat * alpha) / (sqrt(v_hat) + eps);
        beta1_t *= beta1;
        beta2_t *= beta2;
    }
};

template <class Float>
auto factorySGD(Float lr)
{
    return [lr](auto&& W, auto&& dW) { return SGD{W, dW, lr}; };
}

template <class Float>
auto factoryMomentumSGD(Float eta, Float alpha)
{
    return [eta, alpha](auto&& W, auto&& dW) {
        return MomentumSGD{W, dW, eta, alpha};
    };
}

template <class Float>
auto factoryAdam(Float a = 0.001, Float b1 = 0.9, Float b2 = 0.999,
                 Float e = 10e-8)
{
    return [a, b1, b2, e](auto&& W, auto&& dW) {
        return Adam{W, dW, a, b1, b2, e};
    };
}

template <class Optimizers>
void invoke_optimizers(Optimizers& optimizers)
{
    visit_tuple(optimizers, [](auto&& opt) { opt(); });
}

///////////////////////
//// Serialization using library cereal
//////////////////////
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/vector.hpp>
#include <fstream>

namespace ndarray_hpp {
template <class Archive, class Float, size_t... Shape>
void serialize(Archive& ar, ndarray<Float, Shape...>& ary)
{
    ar(ary.data);
}
}  // namespace ndarray_hpp

template <class Archive, class Float, size_t OutSize, size_t... InSize>
void serialize(Archive& ar, Linear<Float, OutSize, InSize...>& src)
{
    ar(src.W, src.b);
}

template <class Archive, class Float, size_t... InSize>
void serialize(Archive& ar, BatchNormalization<Float, InSize...>& src)
{
    ar(src.gamma, src.beta, src.E_x, src.V_x, src.numOfForward);
}

template <class Archive, class Float, size_t... Shape>
void serialize(Archive& ar, Convolution2D<Float, Shape...>& src)
{
    ar(src.We, src.b);
}

// Check if T.serialize(Archive&) or serialize(T, Archive&) exists.
template <class T, class Archive>
using has_serialize_impl =
    decltype(std::declval<T>().serialize(std::declval<Archive&>()));
template <class T, class Archive>
using has_serialize_impl2 =
    decltype(serialize(std::declval<Archive&>(), std::declval<T&>()));
template <class T, class Archive>
using has_serialize =
    std::disjunction<is_detected<has_serialize_impl, T, Archive>,
                     is_detected<has_serialize_impl2, T, Archive>>;

template <class Archive, size_t N = 0, class... Types>
auto layers_serializable(std::tuple<Types...>& stack)
{
    if constexpr (N < sizeof...(Types)) {
        using T = std::decay_t<decltype(std::get<N>(stack))>;
        if constexpr (has_serialize<T, Archive>()) {
            return std::tuple_cat(std::make_tuple(std::ref(std::get<N>(stack))),
                                  layers_serializable<Archive, N + 1>(stack));
        }
        else {
            return layers_serializable<Archive, N + 1>(stack);
        }
    }
    else {
        return std::make_tuple();
    }
}

template <class Archive, class Float, class NetworkStack>
void serialize(Archive& ar, MLP<Float, NetworkStack>& src)
{
    std::apply(ar, layers_serializable<Archive>(src.stack));
}

template <class NN>
void saveNN(const std::string& filepath, const NN& nn, int epoch)
{
    std::ofstream ofs{filepath, std::ios::binary};
    assert(ofs && "Invalid filepath; maybe no permission?");
    cereal::PortableBinaryOutputArchive ar{ofs};
    ar(nn, epoch);
}

template <class NN>
void loadNN(const std::string& filepath, NN& nn, int& epoch)
{
    std::ifstream ifs{filepath, std::ios::binary};
    assert(ifs && "Invalid filepath; maybe not exists?");
    cereal::PortableBinaryInputArchive ar{ifs};
    ar(nn, epoch);
}

/////////////////////
//// Dataset
/////////////////////

#include <mnist/mnist_reader.hpp>
#include <mnist/mnist_utils.hpp>

class MNIST {
public:
    static const size_t N = 60000, NT = 10000;

private:
    const static mnist::MNIST_dataset<std::vector, std::vector<uint8_t>,
                                      uint8_t>
        dataset_;

public:
    template <class Float, size_t BatchSize>
    static std::tuple<Matrix<Float, BatchSize, 28 * 28>,
                      Matrix<Float, BatchSize, 10>>
    train_onehot(const std::array<int, BatchSize>& indices)
    {
        assert(indices.size() == BatchSize && "Invalid indices");

        Matrix<Float, BatchSize, 28 * 28> images;
        for (size_t i = 0; i < BatchSize; i++) {
            auto&& image = dataset_.training_images.at(indices[i]);
            for (size_t j = 0; j < 28 * 28; j++)
                images(i, j) = image.at(j) / 255.;
        }

        auto onehot_labels = Matrix<Float, BatchSize, 10>::zeros();
        for (size_t i = 0; i < BatchSize; i++) {
            int label = dataset_.training_labels.at(indices[i]);
            onehot_labels(i, label) = 1;
        }

        return std::make_tuple(images, onehot_labels);
    }

    template <class Float, size_t BatchSize>
    static std::tuple<Matrix<Float, BatchSize, 28 * 28>, std::vector<int>> test(
        const std::array<int, BatchSize>& indices)
    {
        assert(indices.size() == BatchSize && "Invalid indices");

        Matrix<Float, BatchSize, 28 * 28> images;
        for (size_t i = 0; i < BatchSize; i++) {
            auto&& image = dataset_.test_images.at(indices[i]);
            for (size_t j = 0; j < 28 * 28; j++)
                images(i, j) = image.at(j) / 255.;
        }

        std::vector<int> labels;
        for (auto index : indices)
            labels.push_back(dataset_.test_labels.at(index));

        return std::make_tuple(images, labels);
    }

    template <class Float, size_t BatchSize>
    static std::tuple<ndarray<Float, BatchSize, 1, 28, 28>,
                      Matrix<Float, BatchSize, 10>>
    train_onehot_3d(const std::array<int, BatchSize>& indices)
    {
        // FIXME: transpose
        auto [images, onehot_labels] = train_onehot<Float>(indices);
        return std::make_tuple(images.template reshape<BatchSize, 1, 28, 28>(),
                               onehot_labels);
    }

    template <class Float, size_t BatchSize>
    static std::tuple<ndarray<Float, BatchSize, 1, 28, 28>, std::vector<int>>
    test_3d(const std::array<int, BatchSize>& indices)
    {
        // FIXME: transpose
        auto [images, labels] = test<Float>(indices);
        return std::make_tuple(images.template reshape<BatchSize, 1, 28, 28>(),
                               labels);
    }
};

const mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t>
    MNIST::dataset_ =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();

#include <cifar/cifar10_reader.hpp>

class CIFAR10 {
public:
    static const size_t N = 50000, NT = 10000;

private:
    const static cifar::CIFAR10_dataset<std::vector, std::vector<uint8_t>,
                                        uint8_t>
        dataset_;

public:
    template <class Float, size_t BatchSize>
    static std::tuple<ndarray<Float, BatchSize, 3, 32, 32>,
                      ndarray<Float, BatchSize, 10>>
    train_onehot_3d(const std::array<int, BatchSize>& indices)
    {
        assert(indices.size() == BatchSize && "Invalid indices");

        ndarray<Float, BatchSize, 3, 32, 32> images;
        for (size_t i = 0; i < BatchSize; i++) {
            auto&& image = dataset_.training_images.at(indices[i]);
            for (size_t ch = 0; ch < 3; ch++)
                for (size_t y = 0; y < 32; y++)
                    for (size_t x = 0; x < 32; x++)
                        images(i, ch, y, x) =
                            image.at((ch * 32 + y) * 32 + x) / 255.;
        }

        auto onehot_labels = ndarray<Float, BatchSize, 10>::zeros();
        for (size_t i = 0; i < BatchSize; i++) {
            int label = dataset_.training_labels.at(indices[i]);
            onehot_labels(i, label) = 1;
        }

        return std::make_tuple(images, onehot_labels);
    }

    template <class Float, size_t BatchSize>
    static std::tuple<ndarray<Float, BatchSize, 3, 32, 32>, std::vector<int>>
    test_3d(const std::array<int, BatchSize>& indices)
    {
        assert(indices.size() == BatchSize && "Invalid indices");

        ndarray<Float, BatchSize, 3, 32, 32> images;
        for (size_t i = 0; i < BatchSize; i++) {
            auto&& image = dataset_.test_images.at(indices[i]);
            for (size_t ch = 0; ch < 3; ch++)
                for (size_t y = 0; y < 32; y++)
                    for (size_t x = 0; x < 32; x++)
                        images(i, ch, y, x) =
                            image.at((ch * 32 + y) * 32 + x) / 255.;
        }

        std::vector<int> labels;
        for (auto index : indices)
            labels.push_back(dataset_.test_labels.at(index));

        return std::make_tuple(images, labels);
    }
};
const cifar::CIFAR10_dataset<std::vector, std::vector<uint8_t>, uint8_t>
    CIFAR10::dataset_ =
        cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();

/////////////////////
//// main
/////////////////////

template <size_t M, class T>
std::array<T, M> vec2array(const std::vector<T>& src, size_t offset = 0)
{
    std::array<T, M> ret;
    for (size_t i = 0; i < M; i++)
        ret[i] = src[offset + i];
    return ret;
}

template <class T = void, class... Args>
auto make_array(Args... args)
{
    const size_t size = sizeof...(Args);
    if constexpr (std::is_same<T, void>::value) {
        return std::array<typename std::common_type<Args...>::type, size>{
            args...};
    }
    else {
        return std::array<T, size>{args...};
    }
}

//
#include "range/v3/all.hpp"
using namespace ranges;

template <size_t BatchSize, class Func>
void each_batch(int N, Func func)
{
    auto indices = views::iota(0, (int)N) | to<std::vector> |
                   actions::shuffle(Random::engine());
    for (size_t i = 0; i < N / BatchSize; i++)
        func(i, vec2array<BatchSize>(indices, i * BatchSize));
}

template <class NN>
float train_loss_on_mnist(NN& nn)
{
    auto chosen =
        vec2array<MNIST::N>(views::iota(0, (int)MNIST::N) | to<std::vector>);
    auto [images, onehot_labels] =
        MNIST::train_onehot<typename NN::type>(chosen);
    return nn.forward(images, onehot_labels);
}

template <class NN>
float test_acc_on_mnist(NN& nn)
{
    auto chosen =
        vec2array<MNIST::NT>(views::iota(0, (int)MNIST::NT) | to<std::vector>);
    auto [images, labels] = MNIST::test<typename NN::type>(chosen);
    auto h = nn.predict(images);
    auto res = argmax<1>(h);

    int ncorrect = 0;
    for (size_t i = 0; i < MNIST::NT; i++)
        if (static_cast<int>(res(i)) == labels[i])
            ncorrect++;

    return static_cast<float>(ncorrect) / MNIST::NT;
}

template <class NN>
float train_loss_on_mnist_3d(NN& nn)
{
    static const size_t BatchSize = 100;
    auto indices = views::iota(0, (int)MNIST::N) | to<std::vector>;
    typename NN::type loss_total = 0;
    for (size_t i = 0; i < MNIST::N / BatchSize; i++) {
        auto chosen = vec2array<BatchSize>(indices, i * BatchSize);
        auto [images, onehot_labels] =
            MNIST::train_onehot_3d<typename NN::type>(chosen);
        loss_total += nn.forward(images, onehot_labels);
    }

    return loss_total / (MNIST::N / BatchSize);
}

template <class NN>
float test_acc_on_mnist_3d(NN& nn)
{
    auto chosen =
        vec2array<MNIST::NT>(views::iota(0, (int)MNIST::NT) | to<std::vector>);
    auto [images, labels] = MNIST::test_3d<typename NN::type>(chosen);
    auto h = nn.predict(images);
    auto res = argmax<1>(h);

    int ncorrect = 0;
    for (size_t i = 0; i < MNIST::NT; i++)
        if (static_cast<int>(res(i)) == labels[i])
            ncorrect++;

    return static_cast<float>(ncorrect) / MNIST::NT;
}

template <class NN>
float test_acc_on_cifar_3d(NN& nn)
{
    auto chosen = vec2array<CIFAR10::NT>(views::iota(0, (int)CIFAR10::NT) |
                                         to<std::vector>);
    auto [images, labels] = CIFAR10::test_3d<typename NN::type>(chosen);
    auto h = nn.predict(images);
    auto res = argmax<1>(h);

    int ncorrect = 0;
    for (size_t i = 0; i < CIFAR10::NT; i++)
        if (static_cast<int>(res(i)) == labels[i])
            ncorrect++;

    return static_cast<float>(ncorrect) / CIFAR10::NT;
}

template <class NN>
float train_loss_on_cifar_3d(NN& nn)
{
    static const size_t BatchSize = 100;
    auto indices = views::iota(0, (int)CIFAR10::N) | to<std::vector>;
    typename NN::type loss_total = 0;
    for (size_t i = 0; i < CIFAR10::N / BatchSize; i++) {
        auto chosen = vec2array<BatchSize>(indices, i * BatchSize);
        auto [images, onehot_labels] =
            CIFAR10::train_onehot_3d<typename NN::type>(chosen);
        loss_total += nn.forward(images, onehot_labels);
    }

    return loss_total / (CIFAR10::N / BatchSize);
}
void print_img(const ndarray<float, 1, 28, 28>& src)
{
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            int val = src(0, y, x) * 255;
            std::cout << (char)0x1b << "[48;2;" << val << ";" << val << ";"
                      << val << "m"
                      << "  " << (char)0x1b << "[0m";
        }
        std::cout << std::endl;
    }
}

void print_img(const ndarray<float, 3, 32, 32>& src)
{
    for (int y = 0; y < 32; y++) {
        for (int x = 0; x < 32; x++) {
            int r = src(0, y, x) * 255, g = src(1, y, x) * 255,
                b = src(2, y, x) * 255;
            std::cout << (char)0x1b << "[48;2;" << r << ";" << g << ";" << b
                      << "m"
                      << "  " << (char)0x1b << "[0m";
        }
        std::cout << std::endl;
    }
}

template <size_t BatchSize, class MLP>
void do_with_mlp(const std::string& command, MLP& mlp, int nepochs)
{
    using Float = typename MLP::type;

    int epoch_offset = 0;
    if (command == "cont" || command == "predict" || command == "kadai4")
        loadNN("nn.data", mlp, epoch_offset);

    if (command == "predict") {
        std::cout << "Test acc: " << test_acc_on_mnist(mlp) << std::endl;
        return;
    }

    if (command == "kadai4") {
        int i;
        std::cin >> i;
        auto [image, label] = MNIST::test<Float>(make_array(i));
        auto h = mlp.predict(image);
        std::cout << "Predict:\t" << argmax(h) << std::endl
                  << "Label:\t" << label[0] << std::endl;
        print_img(transpose(std::get<0>(MNIST::test<Float>(make_array(i))))
                      .template reshape<1, 28, 28>());
        return;
    }

    auto optimizers = mlp.optimizers(
        // factorySGD(0.01)
        factoryMomentumSGD(0.01, 0.9)
        // factoryAdam()
    );
    for (int epoch = epoch_offset; epoch < epoch_offset + nepochs; epoch++) {
        each_batch<BatchSize>(MNIST::N, [&](auto index, auto chosen) {
            // Calc acc before step(), where some layers update their statuses.
            Float acc;
            if (index == 0)
                acc = test_acc_on_mnist(mlp);

            // Get training images
            auto [images, onehot_labels] = MNIST::train_onehot<Float>(chosen);
            // Go forward and backward
            Float loss = mlp.step(images, onehot_labels);
            // Update weights
            invoke_optimizers(optimizers);

            // Print loss and acc.
            if (index == 0)
                std::printf(
                    "Epoch: %d\t=> Training loss: %.5f\tTest acc: "
                    "%.4f\n",
                    epoch, loss, acc);
        });
    }
    std::cout << "Last: train loss = " << train_loss_on_mnist(mlp)
              << "\ttest acc = " << test_acc_on_mnist(mlp) << std::endl;

    // Save the result
    saveNN("nn.data", mlp, epoch_offset + nepochs);
}

template <size_t BatchSize, class Conv>
void do_with_conv(const std::string& command, Conv& conv, int nepochs)
{
    using Float = typename Conv::type;

    int epoch_offset = 0;
    if (command == "cont" || command == "predict" || command == "kadai4")
        loadNN("nn_conv.data", conv, epoch_offset);

    if (command == "predict") {
        std::cout << "Test acc: " << test_acc_on_mnist_3d(conv) << std::endl;
        return;
    }

    if (command == "kadai4") {
        int i;
        std::cin >> i;
        auto [image, label] = MNIST::test_3d<Float>(make_array(i));
        auto h = conv.predict(image);
        std::cout << "Predict:\t" << argmax(h) << std::endl
                  << "Label:\t" << label[0] << std::endl;
        print_img(transpose(std::get<0>(MNIST::test<Float>(make_array(i))))
                      .template reshape<1, 28, 28>());
        return;
    }

    auto optimizers = conv.optimizers(
        // factorySGD(0.01)
        factoryMomentumSGD(0.01, 0.9)
        // factoryAdam()//
    );
    for (int epoch = epoch_offset; epoch < epoch_offset + nepochs; epoch++) {
        each_batch<BatchSize>(MNIST::N, [&](auto index, auto chosen) {
            // Calc acc before step(), where some layers update their statuses.
            Float acc;
            if (index == 0)
                acc = test_acc_on_mnist_3d(conv);

            // Get training images
            auto [images, onehot_labels] =
                MNIST::train_onehot_3d<Float>(chosen);
            // Go forward and backward
            Float loss = conv.step(images, onehot_labels);
            // Update weights
            invoke_optimizers(optimizers);

            // Print loss and acc.
            if (index == 0)
                std::printf(
                    "Epoch: %d\t=> Training loss: %.5f\tTest acc: "
                    "%.4f\n",
                    epoch, loss, acc);
        });
    }

    std::cout << "Last: train loss = " << train_loss_on_mnist_3d(conv)
              << "\ttest acc = " << test_acc_on_mnist_3d(conv) << std::endl;

    // Save the result
    saveNN("nn_conv.data", conv, epoch_offset + nepochs);
}

template <class Float>
auto schema_mlp(InputImage<Float, 28 * 28> input)
{
    const static size_t NUnit = 1000, NOut = 10;

    auto h1 = linear<NUnit>(input);
    // auto h2 = dropout(h1);
    auto h2 = batch_normalization(h1);
    auto h3 = relu(h2);
    auto h4 = linear<NOut>(h3);
    auto h5 = softmax_cross_entropy(h4);
    return h5;
}

template <class Float>
auto schema_conv(InputImage<Float, 1, 28, 28> input)
{
    auto h1 = convolution_2d<30, 5, 5, 1, 0>(input);
    auto k1 = batch_normalization(h1);
    auto h2 = relu(k1);
    auto h3 = max_pooling_2d<2, 2, 2, 0>(h2);
    auto h4 = linear<100>(h3);
    auto k0 = batch_normalization(h4);
    auto h5 = relu(k0);
    auto h6 = linear<10>(h5);
    auto h7 = softmax_cross_entropy(h6);

    return h7;
}

template <class Float>
auto schema_conv_cifar(InputImage<Float, 3, 32, 32> input)
{
    auto h1 = convolution_2d<30, 5, 5, 1, 0>(input);
    auto k1 = batch_normalization(h1);
    auto h2 = relu(k1);
    auto h3 = max_pooling_2d<2, 2, 2, 0>(h2);
    auto h4 = linear<100>(h3);
    auto k0 = batch_normalization(h4);
    auto h5 = relu(k0);
    auto h6 = linear<10>(h5);
    auto h7 = softmax_cross_entropy(h6);

    return h7;
}

template <size_t BatchSize, class Conv>
void do_with_cifar(const std::string& command, Conv& conv, int nepochs)
{
    using Float = float;

    int epoch_offset = 0;
    if (command == "cont" || command == "predict" || command == "kadai4")
        loadNN("nn_conv_cifar.data", conv, epoch_offset);

    if (command == "predict") {
        std::cout << "Test acc: " << test_acc_on_cifar_3d(conv) << std::endl;
        return;
    }

    if (command == "kadai4") {
        int i;
        std::cin >> i;
        auto [image, label] = CIFAR10::test_3d<Float>(make_array(i));
        auto h = conv.predict(image);
        std::cout << "Predict:\t" << argmax(h) << std::endl
                  << "Label:\t" << label[0] << std::endl;
        print_img(std::get<0>(CIFAR10::test_3d<Float>(make_array(i)))
                      .template reshape<3, 32, 32>());
        return;
    }

    auto optimizers = conv.optimizers(
        // factorySGD(0.01)
        factoryMomentumSGD(0.01, 0.9)
        // factoryAdam()//
    );
    for (int epoch = epoch_offset; epoch < epoch_offset + nepochs; epoch++) {
        each_batch<BatchSize>(CIFAR10::N, [&](auto index, auto chosen) {
            // Calc acc before step(), where some layers update their statuses.
            Float acc;
            if (index == 0)
                acc = test_acc_on_cifar_3d(conv);

            // Get training images
            auto [images, onehot_labels] =
                CIFAR10::train_onehot_3d<Float>(chosen);
            // Go forward and backward
            Float loss = conv.step(images, onehot_labels);
            // Update weights
            invoke_optimizers(optimizers);

            // Print loss and acc.
            if (index == 0)
                std::printf(
                    "Epoch: %d\t=> Training loss: %.5f\tTest acc: "
                    "%.4f\n",
                    epoch, loss, acc);
        });
    }

    std::cout << "Last: train loss = " << train_loss_on_cifar_3d(conv)
              << "\ttest acc = " << test_acc_on_cifar_3d(conv) << std::endl;

    // Save the result
    saveNN("nn_conv_cifar.data", conv, epoch_offset + nepochs);
}

int main(int argc, char** argv)
{
    using Float = float;

    const static size_t B = 100;
    const static int NEPOCHS = 1;

    assert(2 <= argc && argc <= 3 &&
           "Usage: ./main mlp|conv|cifar [cont|predict|kadai4]");
    const std::string netkind = argv[1];
    const std::string command = [argc, argv]() {
        return argc == 3 ? argv[2] : "";
    }();

    if (netkind == "mlp") {
        auto mlp = define_nn(schema_mlp<Float>);
        do_with_mlp<B>(command, mlp, NEPOCHS);
    }
    else if (netkind == "conv") {
        auto conv = define_nn(schema_conv<Float>);
        do_with_conv<B>(command, conv, NEPOCHS);
    }
    else if (netkind == "cifar") {
        auto conv2 = define_nn(schema_conv_cifar<Float>);
        do_with_cifar<B>(command, conv2, NEPOCHS);
    }
}
