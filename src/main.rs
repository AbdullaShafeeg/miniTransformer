#[derive(Clone, Debug)]
pub struct TensorNode {
    pub shape: Vec<usize>,
    pub value: Vec<f64>,
    pub grad: Vec<f64>,
}

impl TensorNode {
    pub fn new(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        TensorNode {
            shape,
            value: vec![0.0; size],
            grad: vec![0.0; size],
        }
    }

    pub fn zero_grad(&mut self) {
        for g in self.grad.iter_mut() { *g = 0.0; }
    }
}

fn add_forward(a: &TensorNode, b: &TensorNode, out: &mut TensorNode) {
    for i in 0..a.value.len() { out.value[i] = a.value[i] + b.value[i]; }
}

fn add_backward(grad_out: &TensorNode, grad_a: &mut TensorNode, grad_b: &mut TensorNode) {
    for i in 0..grad_out.grad.len() {
        grad_a.grad[i] += grad_out.grad[i];
        grad_b.grad[i] += grad_out.grad[i];
    }
}

fn relu_forward(a: &TensorNode, out: &mut TensorNode) {
    for i in 0..a.value.len() { out.value[i] = a.value[i].max(0.0); }
}

fn relu_backward(a: &TensorNode, grad_out: &TensorNode, grad_a: &mut TensorNode) {
    for i in 0..a.value.len() {
        grad_a.grad[i] += grad_out.grad[i] * if a.value[i] > 0.0 {1.0} else {0.0};
    }
}

fn transpose(a: &TensorNode) -> TensorNode {
    let n = a.shape[0];
    let m = a.shape[1];
    let mut out = TensorNode::new(vec![m, n]);
    for i in 0..n {
        for j in 0..m { out.value[j*n + i] = a.value[i*m + j]; }
    }
    out
}

fn matmul_forward(a: &TensorNode, b: &TensorNode, out: &mut TensorNode) {
    let n = a.shape[0];
    let k = a.shape[1];
    let m = b.shape[1];

    for i in 0..n {
        for j in 0..m {
            let mut sum = 0.0;
            for t in 0..k { sum += a.value[i*k + t] * b.value[t*m + j]; }
            out.value[i*m + j] = sum;
        }
    }
}

fn matmul_backward(
    a: &TensorNode, 
    b: &TensorNode, 
    grad_out: &TensorNode, 
    grad_a: &mut TensorNode, 
    grad_b: &mut TensorNode
) {
    let n = a.shape[0];
    let k = a.shape[1];
    let m = b.shape[1];

    for i in 0..n {
        for t in 0..k {
            let mut sum = 0.0;
            for j in 0..m {
                sum += grad_out.grad[i*m + j] * b.value[t*m + j];
            }
            grad_a.grad[i*k + t] += sum;
        }
    }

    for t in 0..k {
        for j in 0..m {
            let mut sum = 0.0;
            for i in 0..n {
                sum += a.value[i*k + t] * grad_out.grad[i*m + j];
            }
            grad_b.grad[t*m + j] += sum;
        }
    }
}


fn softmax_cross_entropy_loss(logits: &TensorNode, targets: &Vec<usize>) -> f64 {
    let n = logits.shape[0];
    let m = logits.shape[1];
    let mut loss = 0.0;
    for i in 0..n {
        let mut max_val = f64::NEG_INFINITY;
        for j in 0..m { if logits.value[i*m+j] > max_val { max_val = logits.value[i*m+j]; } }
        let mut sum_exp = 0.0;
        for j in 0..m { sum_exp += (logits.value[i*m+j] - max_val).exp(); }
        loss += -logits.value[i*m + targets[i]] + max_val + sum_exp.ln();
    }
    loss / n as f64
}

fn softmax_cross_entropy_backward(logits: &TensorNode, targets: &Vec<usize>, grad_logits: &mut TensorNode) {
    let n = logits.shape[0];
    let m = logits.shape[1];

    for i in 0..n {
        let mut max_val = f64::NEG_INFINITY;
        for j in 0..m {
            if logits.value[i*m + j] > max_val { max_val = logits.value[i*m + j]; }
        }

        let mut sum_exp = 0.0;
        let mut softmax_vals = vec![0.0; m];
        for j in 0..m {
            softmax_vals[j] = (logits.value[i*m + j] - max_val).exp();
            sum_exp += softmax_vals[j];
        }
        for j in 0..m { softmax_vals[j] /= sum_exp; }

        let target_idx = targets[i];
        for j in 0..m {
            grad_logits.grad[i*m + j] += (softmax_vals[j] - if j == target_idx {1.0} else {0.0}) / n as f64;
        }
    }
}

fn softmax_forward(a: &TensorNode, out: &mut TensorNode) {
    let n = a.shape[0];
    let m = a.shape[1];

    for i in 0..n {
        let mut max_val = f64::NEG_INFINITY;
        for j in 0..m {
            if a.value[i*m + j] > max_val { max_val = a.value[i*m + j]; }
        }

        let mut sum_exp = 0.0;
        for j in 0..m {
            out.value[i*m + j] = (a.value[i*m + j] - max_val).exp();
            sum_exp += out.value[i*m + j];
        }

        for j in 0..m { out.value[i*m + j] /= sum_exp; }
    }
}

fn multi_head_attention_forward(x: &TensorNode, Wq: &TensorNode, Wk: &TensorNode, Wv: &TensorNode) -> (TensorNode, TensorNode, TensorNode, TensorNode, TensorNode) {
    let mut Q = TensorNode::new(x.shape.clone());
    let mut K = TensorNode::new(x.shape.clone());
    let mut V = TensorNode::new(x.shape.clone());

    matmul_forward(x, Wq, &mut Q);
    matmul_forward(x, Wk, &mut K);
    matmul_forward(x, Wv, &mut V);

    let K_T = transpose(&K);
    let mut scores = TensorNode::new(vec![Q.shape[0], K_T.shape[1]]);
    matmul_forward(&Q, &K_T, &mut scores);

    let mut attn = TensorNode::new(scores.shape.clone());
    softmax_forward(&scores, &mut attn);

    let mut attn_out = TensorNode::new(x.shape.clone());
    matmul_forward(&attn, &V, &mut attn_out);

    (attn_out, Q, K, V, attn)
}

fn multi_head_attention_backward(
    x: &TensorNode, Wq: &TensorNode, Wk: &TensorNode, Wv: &TensorNode, 
    grad_out: &TensorNode, 
    Q: &TensorNode, K: &TensorNode, V: &TensorNode, attn: &TensorNode,
    grad_x: &mut TensorNode, grad_Wq: &mut TensorNode, grad_Wk: &mut TensorNode, grad_Wv: &mut TensorNode
) {

    let grad_attn_out = grad_out.clone();
    let mut grad_attn = TensorNode::new(attn.shape.clone());
    let mut grad_V   = TensorNode::new(V.shape.clone());
    matmul_backward(attn, V, &grad_attn_out, &mut grad_attn, &mut grad_V);

    let mut grad_scores = TensorNode::new(attn.shape.clone());
    softmax_backward(attn, &grad_attn, &mut grad_scores);

    let mut grad_Q = TensorNode::new(Q.shape.clone());
    let mut grad_K = TensorNode::new(K.shape.clone());
    matmul_backward(Q, &transpose(K), &grad_scores, &mut grad_Q, &mut grad_K);

    matmul_backward(x, Wq, &grad_Q, grad_x, grad_Wq);
    matmul_backward(x, Wk, &grad_K, grad_x, grad_Wk);
    matmul_backward(x, Wv, &grad_V, grad_x, grad_Wv);
}

fn softmax_backward(out: &TensorNode, grad_out: &TensorNode, grad_in: &mut TensorNode) {
    let n = out.shape[0];
    let m = out.shape[1];

    for i in 0..n {
        let out_row_start = i * m;
        let grad_out_row_start = i * m;
        let grad_in_row_start = i * m;

        let mut dot_product = 0.0;
        for j in 0..m {
            dot_product += out.value[out_row_start + j] * grad_out.grad[grad_out_row_start + j];
        }

        for j in 0..m {
            let out_val = out.value[out_row_start + j];
            let grad_out_val = grad_out.grad[grad_out_row_start + j];
            grad_in.grad[grad_in_row_start + j] += out_val * (grad_out_val - dot_product);
        }
    }
}

fn feed_forward_forward(x: &TensorNode, W1: &TensorNode, W2: &TensorNode) -> (TensorNode, TensorNode, TensorNode) {
    let mut ff1 = TensorNode::new(vec![x.shape[0], W1.shape[1]]);
    let mut ff1_relu = TensorNode::new(vec![x.shape[0], W1.shape[1]]);
    let mut ff2 = TensorNode::new(vec![x.shape[0], W2.shape[1]]);

    matmul_forward(x, W1, &mut ff1);
    relu_forward(&ff1, &mut ff1_relu);
    matmul_forward(&ff1_relu, W2, &mut ff2);

    (ff2, ff1, ff1_relu)
}

fn feed_forward_backward(
    x: &TensorNode, W1: &TensorNode, W2: &TensorNode, 
    ff1: &TensorNode, ff1_relu: &TensorNode, 
    grad_out: &TensorNode, 
    grad_x: &mut TensorNode, grad_W1: &mut TensorNode, grad_W2: &mut TensorNode
) {
    let grad_ff2 = grad_out.clone();
    let mut grad_ff1_relu = TensorNode::new(ff1_relu.shape.clone());
    
    matmul_backward(ff1_relu, W2, &grad_ff2, &mut grad_ff1_relu, grad_W2);

    let mut grad_ff1 = TensorNode::new(ff1.shape.clone());
    relu_backward(ff1, &grad_ff1_relu, &mut grad_ff1);

    matmul_backward(x, W1, &grad_ff1, grad_x, grad_W1);
}

fn main() {
    let batch = 1;
    let seq_len = 3;
    let d_model = 4;

    let mut x = TensorNode::new(vec![batch*seq_len, d_model]);
    for i in 0..x.value.len() { x.value[i] = (i as f64 + 1.0)/10.0; }

    let mut Wq = TensorNode::new(vec![d_model,d_model]);
    let mut Wk = TensorNode::new(vec![d_model,d_model]);
    let mut Wv = TensorNode::new(vec![d_model,d_model]);
    let mut W1 = TensorNode::new(vec![d_model,d_model]);
    let mut W2 = TensorNode::new(vec![d_model,d_model]);

    for v in Wq.value.iter_mut() { *v = 0.1; }
    for v in Wk.value.iter_mut() { *v = 0.2; }
    for v in Wv.value.iter_mut() { *v = 0.3; }
    for v in W1.value.iter_mut() { *v = 0.5; }
    for v in W2.value.iter_mut() { *v = 0.6; }

    let mut grad_x = TensorNode::new(x.shape.clone());
    let mut grad_Wq = TensorNode::new(Wq.shape.clone());
    let mut grad_Wk = TensorNode::new(Wk.shape.clone());
    let mut grad_Wv = TensorNode::new(Wv.shape.clone());
    let mut grad_W1 = TensorNode::new(W1.shape.clone());
    let mut grad_W2 = TensorNode::new(W2.shape.clone());

    let (output, Q, K, V, attn) = multi_head_attention_forward(&x, &Wq, &Wk, &Wv);
    let mut attn_res = TensorNode::new(output.shape.clone());
    add_forward(&x, &output, &mut attn_res);

    let (ff_out, ff1, ff1_relu) = feed_forward_forward(&attn_res, &W1, &W2);
    let mut final_out = TensorNode::new(ff_out.shape.clone());
    add_forward(&attn_res, &ff_out, &mut final_out);

    let targets = vec![0,1,2];
    let loss = softmax_cross_entropy_loss(&final_out, &targets);
    println!("Loss: {}", loss);

    let mut grad_final = TensorNode::new(final_out.shape.clone());
    softmax_cross_entropy_backward(&final_out, &targets, &mut grad_final);

    let mut grad_attn_res_from_ff = TensorNode::new(attn_res.shape.clone());
    let mut grad_ff = TensorNode::new(ff_out.shape.clone());
    add_backward(&grad_final, &mut grad_attn_res_from_ff, &mut grad_ff);

    feed_forward_backward(&attn_res, &W1, &W2, &ff1, &ff1_relu, &grad_ff, &mut grad_attn_res_from_ff, &mut grad_W1, &mut grad_W2);
    
    let mut grad_x_from_attn = TensorNode::new(x.shape.clone());
    let mut grad_output = TensorNode::new(output.shape.clone());
    add_backward(&grad_attn_res_from_ff, &mut grad_x_from_attn, &mut grad_output);

    multi_head_attention_backward(&x, &Wq, &Wk, &Wv, &grad_output, &Q, &K, &V, &attn, &mut grad_x_from_attn, &mut grad_Wq, &mut grad_Wk, &mut grad_Wv);
    
    grad_x = grad_x_from_attn;

    println!("Gradients w.r.t input x:");
    for i in 0..grad_x.shape[0] {
        for j in 0..grad_x.shape[1] { print!("{:.4} ", grad_x.grad[i*grad_x.shape[1]+j]); }
        println!();
    }

    println!("Gradients w.r.t Wq:");
    for i in 0..grad_Wq.shape[0] {
        for j in 0..grad_Wq.shape[1] { print!("{:.35} ", grad_Wq.grad[i*grad_Wq.shape[1]+j]); }
        println!();
    }

    println!("Gradients w.r.t k:");
    for i in 0..grad_Wk.shape[0] {
        for j in 0..grad_Wk.shape[1] { print!("{:.35} ", grad_Wk.grad[i*grad_Wk.shape[1]+j]); }
        println!();
    }
}