import os
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def load_pos_embed_from_ckpt(
        ckpt_path: str,
        pos_key: str = "pos_embed.pos_embed"
    ):
    """
    从 checkpoint 中读取 position embedding。

    支持两种保存格式：
    1. torch.save(model.state_dict(), xxx.pth)
    2. torch.save({"model_state_dict": model.state_dict(), ...}, xxx.pth)
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    if pos_key not in state_dict:
        print("Cannot find position embedding key.")
        print("Available keys:")
        for k in state_dict.keys():
            print(k)
        raise KeyError(f"{pos_key} not found in checkpoint.")

    pos_embed = state_dict[pos_key]  # (1, N, D)

    if pos_embed.dim() != 3:
        raise ValueError(
            f"Expected pos_embed shape to be (1, N, D), "
            f"but got {tuple(pos_embed.shape)}"
        )

    return pos_embed


def compute_position_similarity(
        pos_embed: torch.Tensor,
        has_cls_token: bool = True
    ):
    """
    根据 position embedding 自动计算 patch 网格大小，并计算位置之间的 cosine similarity。

    输入：
        pos_embed:
            with cls token:    (1, 65, D)
            without cls token: (1, 64, D)

    返回：
        sim_matrix: (num_patches, num_patches)
        grid_size: int

    例如：
        pos_embed = (1, 65, D)
        去掉 cls token 后是 64 个 patch token
        sqrt(64) = 8
        grid_size = 8
    """
    pos_embed = pos_embed.squeeze(0)  # (N, D)

    if has_cls_token:
        pos_embed = pos_embed[1:]     # (num_patches, D)

    num_patches = pos_embed.shape[0]
    grid_size = int(math.sqrt(num_patches))

    if grid_size * grid_size != num_patches:
        raise ValueError(
            f"num_patches={num_patches} is not a perfect square, "
            f"cannot infer square grid size."
        )

    # cosine similarity
    pos_embed = F.normalize(pos_embed, p=2, dim=1)  # (num_patches, D)
    sim_matrix = pos_embed @ pos_embed.T            # (num_patches, num_patches)

    return sim_matrix, grid_size


def visualize_position_similarity_grid(
        sim_matrix: torch.Tensor,
        grid_size: int,
        save_path: str = "pos_embed_similarity_grid.png"
    ):
    """
    把所有位置的相似度图画成 grid_size × grid_size 大图。

    外层 grid_size × grid_size：
        表示当前 query patch 的位置。

    每个子图内部 grid_size × grid_size：
        表示当前 query patch 和所有 patch 位置的 cosine similarity。
    """
    sim_matrix = sim_matrix.detach().cpu()

    num_patches = grid_size * grid_size

    if sim_matrix.shape != (num_patches, num_patches):
        raise ValueError(
            f"Expected sim_matrix shape to be "
            f"({num_patches}, {num_patches}), "
            f"but got {tuple(sim_matrix.shape)}"
        )

    fig, axes = plt.subplots(
        grid_size,
        grid_size,
        figsize=(2.2 * grid_size, 2.2 * grid_size)
    )

    # 当 grid_size=1 时，axes 不是二维数组，这里做兼容
    if grid_size == 1:
        axes = [[axes]]

    im = None

    for idx in range(num_patches):
        row = idx // grid_size
        col = idx % grid_size

        sim_map = sim_matrix[idx].reshape(grid_size, grid_size)

        ax = axes[row][col]
        im = ax.imshow(sim_map, vmin=-1.0, vmax=1.0)

        # 标记当前 query patch 的位置
        ax.scatter(col, row, marker="x", s=20)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"({row},{col})", fontsize=7)

    fig.suptitle(
        "Position Embedding Cosine Similarity Maps",
        fontsize=18
    )

    fig.tight_layout()

    # 添加统一 colorbar
    if im is not None:
        all_axes = [axes[r][c] for r in range(grid_size) for c in range(grid_size)]
        cbar = fig.colorbar(
            im,
            ax=all_axes,
            shrink=0.6
        )
        cbar.set_label("Cosine Similarity")

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved to: {save_path}")


def main():
    ckpt_path = "/home/liujiang/projects/dl/model_re/checkpoints/ViT/vit_base_run_d24/weights/model_latest.pth"
    save_path = os.path.join(
        os.path.dirname(os.path.dirname(ckpt_path)),
        "pos_embed_similarity_grid.png"
    )

    pos_embed = load_pos_embed_from_ckpt(
        ckpt_path=ckpt_path,
        pos_key="pos_embed.pos_embed"
    )

    sim_matrix, grid_size = compute_position_similarity(
        pos_embed=pos_embed,
        has_cls_token=True
    )

    print(f"pos_embed shape: {tuple(pos_embed.shape)}")
    print(f"sim_matrix shape: {tuple(sim_matrix.shape)}")
    print(f"inferred grid_size: {grid_size} x {grid_size}")

    visualize_position_similarity_grid(
        sim_matrix=sim_matrix,
        grid_size=grid_size,
        save_path=save_path
    )


if __name__ == "__main__":
    main()