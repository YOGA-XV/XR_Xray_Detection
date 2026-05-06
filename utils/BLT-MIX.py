import csv
import os
import random
import shutil
from collections import Counter

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
from tqdm import tqdm


def robust_norm(x, eps=1e-6):
    q1 = torch.quantile(x.flatten(), 0.01)
    q99 = torch.quantile(x.flatten(), 0.99)
    x = (x - q1) / (q99 - q1 + eps)
    return x.clamp(0.0, 1.0)


def rgb_to_luminance(img):
    r, g, b = img[0:1], img[1:2], img[2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def BLT_Mix(base_img, src_patch, mask, alpha=0.7):
    Tb = robust_norm(rgb_to_luminance(base_img))
    Ts = robust_norm(rgb_to_luminance(src_patch))
    Tmix = mask * (Tb * Ts.clamp(min=1e-6).pow(alpha)) + (1 - mask) * Tb
    Tmix = Tmix.clamp(0.0, 1.0)
    scale = Tmix / Tb.clamp(min=1e-6)
    mixed_img = (base_img * scale).clamp(0.0, 1.0)

    return mixed_img


def make_transformed_patch(src_img, patch_size, rotate=0, flip_h=False, flip_v=False):
    w_patch, h_patch = patch_size
    patch = src_img
    if flip_h:
        patch = T.functional.hflip(patch)
    if flip_v:
        patch = T.functional.vflip(patch)
    if rotate:
        patch = T.functional.rotate(patch, -rotate)
    return T.functional.resize(patch, (h_patch, w_patch))


def paste_patch_canvas(src_patch, out_size, patch_pos):
    W_out, H_out = out_size
    x_offset, y_offset = patch_pos
    h_patch, w_patch = src_patch.shape[-2:]
    canvas = torch.zeros((src_patch.shape[0], H_out, W_out), dtype=src_patch.dtype, device=src_patch.device)
    canvas[:, y_offset:y_offset + h_patch, x_offset:x_offset + w_patch] = src_patch
    return canvas


def mix_patch_into_base(base_img, src_patch, patch_pos, alpha):
    x_offset, y_offset = patch_pos
    h_patch, w_patch = src_patch.shape[-2:]
    mixed_img = base_img.clone()
    base_roi = base_img[:, y_offset:y_offset + h_patch, x_offset:x_offset + w_patch]
    roi_mask = torch.ones(1, h_patch, w_patch, dtype=base_img.dtype, device=base_img.device)
    mixed_roi = BLT_Mix(base_roi, src_patch, roi_mask, alpha)
    mixed_img[:, y_offset:y_offset + h_patch, x_offset:x_offset + w_patch] = mixed_roi
    return mixed_img


def sample_patch_size(W_out, H_out, min_ratio=0.45, max_ratio=0.75):
    min_w = max(1, int(W_out * min_ratio))
    max_w = max(min_w, int(W_out * max_ratio))
    min_h = max(1, int(H_out * min_ratio))
    max_h = max(min_h, int(H_out * max_ratio))
    return random.randint(min_w, max_w), random.randint(min_h, max_h)


def transform_bboxes(bboxes, patch_pos, patch_size, out_size, rotate=0, flip_h=False, flip_v=False):
    """
    Convert YOLO boxes from a full source image into the transformed patch location.

    bboxes: [[cls, xc, yc, w, h], ...] normalized in source image [0,1]
    patch_pos: (x_offset, y_offset) in output image pixels
    patch_size: (w_patch, h_patch) in output pixels
    out_size: (W,H)
    rotate: 0, 90, 180, 270 degrees clockwise
    """
    W_out, H_out = out_size
    x_offset, y_offset = patch_pos
    w_patch, h_patch = patch_size
    new_bboxes = []

    for b in bboxes:
        cls, xc, yc, w, h = b

        if flip_h:
            xc = 1.0 - xc
        if flip_v:
            yc = 1.0 - yc

        if rotate == 90:
            xc, yc = 1.0 - yc, xc
            w, h = h, w
        elif rotate == 180:
            xc, yc = 1.0 - xc, 1.0 - yc
        elif rotate == 270:
            xc, yc = yc, 1.0 - xc
            w, h = h, w

        x_pixel = x_offset + xc * w_patch
        y_pixel = y_offset + yc * h_patch
        w_pixel = w * w_patch
        h_pixel = h * h_patch

        new_bboxes.append([
            cls,
            x_pixel / W_out,
            y_pixel / H_out,
            w_pixel / W_out,
            h_pixel / H_out,
        ])

    return new_bboxes


def yolo_to_xyxy(bbox, width, height):
    _, xc, yc, w, h = bbox
    x1 = int((xc - w / 2) * width)
    y1 = int((yc - h / 2) * height)
    x2 = int((xc + w / 2) * width)
    y2 = int((yc + h / 2) * height)
    return max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)


def draw_bboxes(image, bboxes, color=(255, 0, 0)):
    drawn = image.copy()
    draw = ImageDraw.Draw(drawn)
    width, height = drawn.size
    for bbox in bboxes:
        cls = int(bbox[0])
        x1, y1, x2, y2 = yolo_to_xyxy(bbox, width, height)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        draw.text((x1 + 2, max(0, y1 - 12)), str(cls), fill=color)
    return drawn


def add_caption(image, caption):
    caption_h = 24
    canvas = Image.new("RGB", (image.width, image.height + caption_h), (245, 245, 245))
    canvas.paste(image, (0, caption_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), caption, fill=(20, 20, 20))
    return canvas


def save_preview_image(base_img, src_canvas, mixed_img, base_bboxes, transformed_bboxes, merged_bboxes, out_path):
    to_pil = T.ToPILImage()
    base_panel = add_caption(draw_bboxes(to_pil(base_img), base_bboxes, color=(0, 180, 255)), "base")
    src_panel = add_caption(draw_bboxes(to_pil(src_canvas), transformed_bboxes, color=(255, 160, 0)), "source patch")
    mixed_panel = add_caption(draw_bboxes(to_pil(mixed_img), merged_bboxes, color=(255, 0, 0)), "mixed")

    gap = 8
    width = base_panel.width + src_panel.width + mixed_panel.width + gap * 2
    height = max(base_panel.height, src_panel.height, mixed_panel.height)
    preview = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for panel in (base_panel, src_panel, mixed_panel):
        preview.paste(panel, (x, 0))
        x += panel.width + gap
    preview.save(out_path, quality=95)


def read_yolo_labels(label_file):
    with open(label_file, encoding="utf-8") as f:
        return [list(map(float, line.strip().split())) for line in f if line.strip()]


def count_classes_from_bboxes(bboxes):
    counts = Counter()
    for bbox in bboxes:
        counts[int(bbox[0])] += 1
    return counts


def count_classes_in_label_dir(label_dir):
    counts = Counter()
    for label_name in os.listdir(label_dir):
        if not label_name.lower().endswith(".txt"):
            continue
        counts.update(count_classes_from_bboxes(read_yolo_labels(os.path.join(label_dir, label_name))))
    return counts


def write_class_distribution_csv(out_path, original_counts, augmented_counts):
    class_ids = sorted(set(original_counts) | set(augmented_counts))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "original_count", "augmented_count", "delta"])
        writer.writeheader()
        for class_id in class_ids:
            original = original_counts.get(class_id, 0)
            augmented = augmented_counts.get(class_id, 0)
            writer.writerow({
                "class_id": class_id,
                "original_count": original,
                "augmented_count": augmented,
                "delta": augmented - original,
            })


def offline_BLT_Mix_with_labels_v2(
    img_dir,
    label_dir,
    out_dir,
    out_size=(512, 512),
    n_aug_per_image=2,
    alpha_range=(0.4, 1.0),
    preview_count=8,
):
    os.makedirs(out_dir, exist_ok=True)
    out_img_dir = os.path.join(out_dir, "images")
    out_label_dir = os.path.join(out_dir, "labels")
    preview_dir = os.path.join(out_dir, "preview")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)
    if preview_count > 0:
        os.makedirs(preview_dir, exist_ok=True)

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    to_tensor = T.ToTensor()
    to_pil = T.ToPILImage()

    W_out, H_out = out_size
    previews_written = 0
    augmented_images = 0
    original_class_counts = count_classes_in_label_dir(label_dir)
    output_class_counts = Counter()

    print("BLT-Mix configuration")
    print(f"  Image dir: {img_dir}")
    print(f"  Label dir: {label_dir}")
    print(f"  Output dir: {out_dir}")
    print(f"  Output size: {W_out}x{H_out}")
    print(f"  Source images: {len(img_files)}")
    print(f"  Augmentations per image: {n_aug_per_image}")
    print(f"  Preview limit: {preview_count}")

    for img_file in tqdm(img_files, desc="BLT-Mix images", unit="img"):
        base_img_path = os.path.join(img_dir, img_file)
        base_img = to_tensor(Image.open(base_img_path).convert("RGB"))
        base_img = T.functional.resize(base_img, out_size)

        label_file = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")
        base_bboxes_scaled = read_yolo_labels(label_file)
        output_class_counts.update(count_classes_from_bboxes(base_bboxes_scaled))

        shutil.copy(base_img_path, os.path.join(out_img_dir, img_file))
        shutil.copy(label_file, os.path.join(out_label_dir, os.path.basename(label_file)))

        for i in range(n_aug_per_image):
            src_file = random.choice(img_files)
            src_img = to_tensor(Image.open(os.path.join(img_dir, src_file)).convert("RGB"))
            src_img = T.functional.resize(src_img, out_size)

            src_label_file = os.path.join(label_dir, os.path.splitext(src_file)[0] + ".txt")
            src_bboxes = read_yolo_labels(src_label_file)

            w_patch, h_patch = sample_patch_size(W_out, H_out)
            x_off = random.randint(0, W_out - w_patch)
            y_off = random.randint(0, H_out - h_patch)

            alpha = random.uniform(alpha_range[0], alpha_range[1])
            rotate = random.choice([0, 90, 180, 270])
            flip_h = random.choice([True, False])
            flip_v = random.choice([True, False])

            src_patch = make_transformed_patch(
                src_img,
                (w_patch, h_patch),
                rotate=rotate,
                flip_h=flip_h,
                flip_v=flip_v,
            )

            mixed_img = mix_patch_into_base(base_img, src_patch, (x_off, y_off), alpha)

            transformed_bboxes = transform_bboxes(
                src_bboxes,
                (x_off, y_off),
                (w_patch, h_patch),
                out_size,
                rotate,
                flip_h,
                flip_v,
            )
            merged_bboxes = base_bboxes_scaled + transformed_bboxes
            output_class_counts.update(count_classes_from_bboxes(merged_bboxes))

            out_name = f"{os.path.splitext(img_file)[0]}_BLT{i}.png"
            to_pil(mixed_img).save(os.path.join(out_img_dir, out_name))
            augmented_images += 1

            out_label_file = os.path.join(out_label_dir, f"{os.path.splitext(img_file)[0]}_BLT{i}.txt")
            with open(out_label_file, "w", encoding="utf-8") as f:
                for b in merged_bboxes:
                    f.write(" ".join(map(str, b)) + "\n")

            if preview_count > 0 and previews_written < preview_count:
                src_canvas = paste_patch_canvas(src_patch, out_size, (x_off, y_off))
                preview_name = f"{os.path.splitext(img_file)[0]}_BLT{i}_preview.jpg"
                save_preview_image(
                    base_img,
                    src_canvas,
                    mixed_img,
                    base_bboxes_scaled,
                    transformed_bboxes,
                    merged_bboxes,
                    os.path.join(preview_dir, preview_name),
                )
                previews_written += 1

    distribution_path = os.path.join(out_dir, "class_distribution.csv")
    write_class_distribution_csv(distribution_path, original_class_counts, output_class_counts)

    print("BLT-Mix summary")
    print(f"  Original images copied: {len(img_files)}")
    print(f"  Augmented images: {augmented_images}")
    print(f"  Preview images: {previews_written}")
    print("  Class distribution:")
    for class_id in sorted(set(original_class_counts) | set(output_class_counts)):
        original = original_class_counts.get(class_id, 0)
        augmented = output_class_counts.get(class_id, 0)
        print(f"    class {class_id}: original={original}, augmented={augmented}, delta={augmented - original}")
    print(f"  Class distribution saved to: {distribution_path}")
    print(f"  Output saved to: {out_dir}")


if __name__ == "__main__":
    img_dir = "./datasets/SPXray/images/val"
    label_dir = "./datasets/SPXray/labels/val"
    out_dir = "./datasets/SPXray_BLT_Mix/val"

    offline_BLT_Mix_with_labels_v2(
        img_dir,
        label_dir,
        out_dir,
        out_size=(640, 640),
        n_aug_per_image=1,
        preview_count=30,
    )
