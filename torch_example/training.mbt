///|
pub(all) struct TrainingResult {
  losses : Array[Double]
  train_accuracy : Double
  test_accuracy : Double
  device : String
}

///|
pub fn neural_network(
  num_inputs~ : Int32,
  num_outputs~ : Int32,
) -> @torch.TorchModule raise @torch.TorchError {
  @torch.sequential([
    @torch.linear(in_features=num_inputs, out_features=30),
    @torch.relu(),
    @torch.linear(in_features=30, out_features=20),
    @torch.relu(),
    @torch.linear(in_features=20, out_features=num_outputs),
  ])
}

///|
fn toy_data() -> (@torch.Tensor, @torch.Tensor, @torch.Tensor, @torch.Tensor) raise @torch.TorchError {
  let x_train = @torch.tensor(
    data=[-1.2, 3.1, -0.9, 2.9, -0.5, 2.6, 2.3, -1.1, 2.7, -1.5],
    shape=[5, 2],
  )
  let y_train = @torch.int_tensor(data=[0L, 0L, 0L, 1L, 1L], shape=[5])
  let x_test = @torch.tensor(data=[-0.8, 2.8, 2.6, -1.6], shape=[2, 2])
  let y_test = @torch.int_tensor(data=[0L, 1L], shape=[2])
  (x_train, y_train, x_test, y_test)
}

///|
fn loader(
  features : @torch.Tensor,
  labels : @torch.Tensor,
  shuffle~ : Bool,
  drop_last~ : Bool,
) -> @torch.TorchDataLoader raise @torch.TorchError {
  @torch.data_loader(
    dataset=@torch.tensor_dataset(features~, labels~),
    batch_size=2,
    shuffle~,
    num_workers=0,
    drop_last~,
  )
}

///|
pub fn compute_accuracy(
  model : @torch.TorchModule,
  data_loader : @torch.TorchDataLoader,
  device~ : String,
) -> Double raise @torch.TorchError {
  model.eval()
  let mut correct = 0.0
  let mut total = 0.0
  for batch in data_loader.batches() {
    let features = batch.features.to(device~)
    let labels = batch.labels.to(device~)
    let logits = model.forward(input=features)
    let predictions = @torch.argmax(input=logits, dim=1).to_array()
    let expected = labels.to_array()
    for index, label in expected {
      if predictions[index] == label {
        correct += 1.0
      }
      total += 1.0
    }
  }
  correct / total
}

///|
pub fn train_toy_classifier(
  device~ : String = "cpu",
  epochs~ : Int32 = 3,
) -> TrainingResult raise @torch.TorchError {
  let (x_train, y_train, x_test, y_test) = toy_data()
  let train_loader = loader(x_train, y_train, shuffle=true, drop_last=true)

  @torch.manual_seed(seed=123L)
  let model = neural_network(num_inputs=2, num_outputs=2)
  model.to(device~)
  let optimizer = @torch.sgd(parameters=model.parameters(), learning_rate=0.5)
  let losses : Array[Double] = []

  for _epoch in 0..<epochs {
    model.train()
    for batch in train_loader.batches() {
      let features = batch.features.to(device~)
      let labels = batch.labels.to(device~)
      let logits = model.forward(input=features)
      let loss = @torch.cross_entropy(input=logits, target=labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      losses.push(loss.item())
    }
    model.eval()
  }

  let train_evaluation_loader = loader(
    x_train,
    y_train,
    shuffle=false,
    drop_last=false,
  )
  let test_loader = loader(x_test, y_test, shuffle=false, drop_last=false)
  TrainingResult::{
    losses,
    train_accuracy: compute_accuracy(model, train_evaluation_loader, device~),
    test_accuracy: compute_accuracy(model, test_loader, device~),
    device,
  }
}
