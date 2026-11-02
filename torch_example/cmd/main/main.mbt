///|
fn main {
  let result = @torch_example.train_toy_classifier() catch { _ => panic() }
  println("device: \{result.device}")
  for index, loss in result.losses {
    println("batch \{index} loss: \{loss}")
  }
  println("train accuracy: \{result.train_accuracy}")
  println("test accuracy: \{result.test_accuracy}")
}
