from pytorch_tabnet.tab_network import TabNet
import paddle


if __name__ == "__main__":
    cat_dims = [73, 9, 16, 16, 7, 15, 6, 5, 2, 119, 92, 94, 42]
    cat_idxs = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    cat_emb_dim = [2] * 13
    group_attention_matrix = paddle.rand((10, 14))
    model = TabNet(14,
                   2,
                   cat_dims=cat_dims,
                   cat_idxs=cat_idxs,
                   cat_emb_dim=cat_emb_dim,
                   group_attention_matrix=group_attention_matrix)
    x = paddle.rand((1024, 14))
    try:
        x = paddle.static.InputSpec.from_tensor(x)
        paddle.jit.save(model, input_spec=(x,), path="./model")
        print("[JIT] paddle.jit.save successed.")
        exit(0)
    except Exception as e:
        print("[JIT] paddle.jit.save failed.")
        raise e
