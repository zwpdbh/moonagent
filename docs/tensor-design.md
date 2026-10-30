# Tensor design principles

## Purpose

The `zw/tensor` package owns generic numerical data structures and operations.
Qwen-specific architecture and inference orchestration remain in `qwen3`.
The design favors a correct, readable CPU reference implementation before
optimization.

## Flat storage, explicit shape

Tensors use contiguous row-major `Float` storage. Vectors, matrices, and
higher-rank tensors share this representation because it matches safetensors,
avoids nested allocations, and fits CPU/GPU buffers.

Shape is part of the tensor abstraction rather than a separate argument at every
call site. Checked constructors reject negative dimensions and data lengths that
do not equal the product of the shape dimensions.

## Owned tensors and borrowed views

`Tensor` owns an `Array[Float]` and is used for newly computed results and stored
weights. `TensorView` borrows an `ArrayView[Float]` with its shape and is used for
read-only inputs and zero-copy contiguous slices.

Operations return owned tensors. Views are temporary borrow boundaries; they are
not computation-graph nodes. Tensors are immutable by default, with mutation
confined to clearly owned kernels and cache-management code.

## API boundaries

Public model-level operations accept and return `Tensor` or `TensorView`.
Low-level kernels may still use flat arrays where rank and layout are already
validated. Labeled arguments clarify operand roles, especially when vectors and
matrices share the same underlying representation.

Shape validation belongs at operation boundaries. Implementations derive
dimensions from shapes rather than requiring redundant size parameters.

## Arithmetic operators

Named functions are the canonical operation implementations. Owned `Tensor`
values also support `+`, `-`, `*`, and `/` as thin convenience wrappers. All four
operators are element-wise, preserve shape, and require exactly equal shapes;
there is no implicit broadcasting.

In particular, `*` never means matrix multiplication. Matrix/vector products and
other operations with different shape semantics keep explicit names such as
`mat_vec_mul` and `vec_mat_mul`. Operators make residual arithmetic concise
without hiding the meaning of more complex operations.

## Incremental feature growth

The initial implementation supports contiguous row-major tensors only. Strides,
broadcasting, and non-contiguous views will be added when concrete operations
require them, rather than speculatively building a complete tensor framework.

## Future GPU execution

Flat storage and centralized numerical kernels leave room for alternate execution
backends. Backend/device abstractions will be introduced when GPU work begins;
model-level code should depend on tensor operations rather than kernel details.

## Future training and autograd

`Tensor` deliberately contains only forward data and shape. Autograd state does
not belong in the inference tensor.

A future training layer can define an owning wrapper (for example, `Variable`)
with gradients, `requires_grad`, and computation-graph metadata. Forward kernels
can be reused while backward rules and graph management are implemented
separately. Graph nodes must own or share stable values rather than retain
transient `TensorView` inputs.
