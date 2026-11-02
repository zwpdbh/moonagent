///|
test "article autograd computes weight and bias gradients" {
  let y = @torch.tensor(data=[1.0], shape=[1])
  let x1 = @torch.tensor(data=[1.1], shape=[1])
  let w1 = @torch.tensor(data=[2.2], shape=[1], requires_grad=true)
  let b = @torch.tensor(data=[0.0], shape=[1], requires_grad=true)

  let z = @torch.add(left=@torch.mul(left=x1, right=w1), right=b)
  let a = @torch.sigmoid(input=z)
  let loss = @torch.binary_cross_entropy(input=a, target=y)
  loss.backward()

  assert_true((loss.item() - 0.0852).abs() < 0.0001)
  assert_true((w1.grad().item() + 0.0898).abs() < 0.0001)
  assert_true((b.grad().item() + 0.0817).abs() < 0.0001)
}

///|
test "article training loop converges on the toy dataset" {
  let result = @torch_example.train_toy_classifier()
  assert_eq(result.losses.length(), 6)
  assert_true(result.losses[result.losses.length() - 1] < result.losses[0])
  assert_eq(result.train_accuracy, 1.0)
  assert_eq(result.test_accuracy, 1.0)
}

///|
test "article tensor addition runs on CUDA when available" {
  if @torch.cuda_is_available() {
    let left = @torch.tensor(data=[1.0, 2.0, 3.0], shape=[3]).to(device="cuda")
    let right = @torch.tensor(data=[4.0, 5.0, 6.0], shape=[3]).to(device="cuda")
    let result = @torch.add(left~, right~)
    assert_eq(result.device(), "cuda:0")
    assert_eq(result.to_array(), [5.0, 7.0, 9.0])
  }
}

///|
test "article training loop runs on CUDA when available" {
  if @torch.cuda_is_available() {
    let result = @torch_example.train_toy_classifier(device="cuda", epochs=1)
    assert_eq(result.device, "cuda")
    assert_eq(result.losses.length(), 2)
  }
}
