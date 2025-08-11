from PIL import Image, ImageDraw, ImageFilter
import io

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

model = None
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

app = FastAPI()


def detect_objects_with_bbox(image: Image.Image):
    image = np.array(image.convert("RGB"))
    masks = model.generate(image)
    bounding_boxes = [mask['bbox'] for mask in masks]
    return bounding_boxes


def generate_instance_segmented_image(image: Image.Image):
    input_img = np.array(image.convert("RGB"))
    masks = model.generate(input_img)
    
    np.random.seed(3)
    for mask in masks:
        color = (255 * np.random.random(3)).astype(np.uint8)
        segmentation = mask['segmentation']
        mask = Image.fromarray(segmentation.astype(np.uint8) * 128)
        edged_mask = Image.blend(mask, mask.filter(ImageFilter.EDGE_ENHANCE), 0.5)
        color_img = Image.fromarray(np.ones_like(input_img) * color[None, None, :])
        image = Image.composite(color_img, image, edged_mask)
        
    draw = ImageDraw.Draw(image)
    np.random.seed(3)
    for mask in masks:
        color = (255 * np.random.random(3)).astype(np.uint8)
        x, y, w, h = mask['bbox']
        draw.rectangle([(x, y), (x + w, y + h)], outline=tuple(color), width=3)
        
    return image


@app.on_event("startup")
async def startup_event():
    global model, device

    if "cuda" in device:
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    model = SAM2AutomaticMaskGenerator(
        sam2,
        points_per_side = 32,
        points_per_batch = 64,
        pred_iou_thresh = 0.8,
        stability_score_thresh = 0.75,
        stability_score_offset = 1.0,
        mask_threshold = 0.0,
        box_nms_thresh = 0.7,
        crop_n_layers = 0,
        crop_nms_thresh = 0.7,
        crop_overlap_ratio = 512 / 1500,
        crop_n_points_downscale_factor = 1,
        point_grids = None,
        min_mask_region_area = 0,
        output_mode = "binary_mask",
        use_m2m = False,
        multimask_output = True,
    )
    print("FastAPIアプリケーションが起動しました。")


@app.post("/detect/")
async def detect_bounding_boxes(file: UploadFile = File(...)):
    # アップロードされたファイルが画像であることを確認
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="無効なファイル形式です。画像をアップロードしてください。"
        )

    try:
        # ファイルの内容をメモリに読み込み
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 物体検出を実行
        bounding_boxes = detect_objects_with_bbox(image)
        
        # 結果をJSON形式で返す
        return JSONResponse(content={"bounding_boxes": bounding_boxes})
        
    except Exception as e:
        # エラーハンドリング
        raise HTTPException(status_code=500, detail=f"処理中にエラーが発生しました: {e}")
    
    
@app.post("/detect-demo/")
async def detect_bounding_boxes_demo(file: UploadFile = File(...)):
    # アップロードされたファイルが画像であることを確認
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="無効なファイル形式です。画像をアップロードしてください。"
        )

    try:
        # ファイルの内容をメモリに読み込み
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 物体検出を実行
        output_img = generate_instance_segmented_image(image)

        # 描画された画像をJPEG形式でバイト列に変換
        img_byte_io = io.BytesIO()
        output_img.save(img_byte_io, format="JPEG")
        img_byte_io.seek(0)

        # 画像ストリームをHTTPレスポンスとして返す
        return StreamingResponse(img_byte_io, media_type="image/jpeg")
            
        
    except Exception as e:
        # エラーハンドリング
        raise HTTPException(status_code=500, detail=f"処理中にエラーが発生しました: {e}")