
from megatron.core.migrated_lazy import LazyCall as L
from megatron.core.models.vision.dfn_rope import ViT
from megatron.training import get_args

args = get_args()

meta = dict(
    vision_checkpoint="/path/to/dfn_huge_378.pth",
)

vision_model = L(ViT)(
    img_size=378,
    patch_size=args.spatial_patch_size,
    temporal_patch_size=args.temporal_patch_size,
    spatial_merge_size=args.spatial_merge_size,
    temporal_merge_size=args.temporal_merge_size,
    in_chans=3,
    num_classes=args.hidden_size,
    embed_dim=1280,
    depth=32,
    num_heads=16,
    mlp_ratio=4,
    grad_checkpointing=args.vision_grad_checkpointing,
    enable_mp=args.vit_enable_mp,
    vision_sequence_parallel=args.vision_sequence_parallel,
)
