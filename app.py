import os
from flask import Flask, render_template, request
import cv2
import numpy as np
from rembg import remove
from PIL import Image
import io
import subprocess
import re

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ------------------------- 画像処理関数 -------------------------
def process_image(image_stream, size=(30, 30)):
    input_bytes = image_stream.read()
    out_bytes = remove(input_bytes)
    img_pil = Image.open(io.BytesIO(out_bytes)).convert("RGBA")
    img_np_rgba = np.array(img_pil)
    img_np_rgb = img_np_rgba[:, :, :3]
    img_np_gray = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2GRAY)
    alpha_channel = img_np_rgba[:, :, 3]
    object_mask = (alpha_channel > 0).astype(np.uint8) * 255
    img_masked_gray = np.where(object_mask == 255, img_np_gray, 255)
    img_blur = cv2.medianBlur(img_masked_gray, 5)
    img_binary = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
    img_resized_gray = cv2.resize(img_binary, size, interpolation=cv2.INTER_AREA)
    img_processed = cv2.adaptiveThreshold(img_resized_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'processed_image.png'), img_processed)
    return img_processed

# ------------------------- ヒント計算 -------------------------
def get_hints(grid):
    row_hints = calculate_line_hints(grid)
    col_hints = calculate_line_hints(list(zip(*grid)))
    return row_hints, col_hints

def calculate_line_hints(lines):
    hints = []
    for line in lines:
        row_hint = []
        count = 0
        for cell in line:
            if cell == 1:
                count += 1
            elif count > 0:
                row_hint.append(count)
                count = 0
        if count > 0:
            row_hint.append(count)
        if not row_hint:
            row_hint.append(0)
        hints.append(row_hint)
    return hints

# ------------------------- CSP生成 (位置変数方式) -------------------------
def generate_csp_file(row_hints, col_hints, width, height, csp_filename):
    try:
        with open(csp_filename, 'w') as f:
            # 1. 各マス変数
            for y in range(height):
                for x in range(width):
                    f.write(f"(int x_{y}_{x} 0 1)\n")
            f.write("\n")

            # 2. 行ブロック変数
            for y, hints in enumerate(row_hints):
                if hints == [0]:
                    continue
                for i, size in enumerate(hints):
                    max_pos = width - size
                    f.write(f"(int h_{y}_{i} 0 {max_pos})\n")
            f.write("\n")

            # 3. 列ブロック変数
            for x, hints in enumerate(col_hints):
                if hints == [0]:
                    continue
                for i, size in enumerate(hints):
                    max_pos = height - size
                    f.write(f"(int v_{x}_{i} 0 {max_pos})\n")
            f.write("\n")

            # 4. 行ブロックの順序制約（間隔1以上）
            for y, hints in enumerate(row_hints):
                for i in range(len(hints)-1):
                    f.write(f"(<= (+ h_{y}_{i} {hints[i]} 1) h_{y}_{i+1})\n")
            # 5. 列ブロックの順序制約
            for x, hints in enumerate(col_hints):
                for i in range(len(hints)-1):
                    f.write(f"(<= (+ v_{x}_{i} {hints[i]} 1) v_{x}_{i+1})\n")

            f.write("\n")
            
            # 6. 行マス変数とブロック位置の関係
            for y, hints in enumerate(row_hints):
                if hints == [0]:
                    for x in range(width):
                        f.write(f"(= x_{y}_{x} 0)\n")
                    continue
                for x in range(width):
                    clauses = []
                    for i, size in enumerate(hints):
                        clauses.append(f"(and (<= h_{y}_{i} {x}) (< {x} (+ h_{y}_{i} {size})))")
                    f.write(f"(iff (= x_{y}_{x} 1) (or {' '.join(clauses)}))\n")

            # 7. 列マス変数とブロック位置の関係
            for x, hints in enumerate(col_hints):
                if hints == [0]:
                    for y in range(height):
                        f.write(f"(= x_{y}_{x} 0)\n")
                    continue
                for y in range(height):
                    clauses = []
                    for i, size in enumerate(hints):
                        clauses.append(f"(and (<= v_{x}_{i} {y}) (< {y} (+ v_{x}_{i} {size})))")
                    f.write(f"(iff (= x_{y}_{x} 1) (or {' '.join(clauses)}))\n")

        return True

    except Exception as e:
        import traceback
        traceback.print_exc()
        return False

# ------------------------- 一意性チェック -------------------------
def check_unicity(row_hints, col_hints, width, height, csp_filename):
    if not generate_csp_file(row_hints, col_hints, width, height, csp_filename):
        return "Error: CSPファイル生成失敗"
    try:
        os.environ["PATH"] += ":/home/takana/projects/jottoku/sugar-2.3.4/bin"
        result1 = subprocess.run(["sugar", csp_filename],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out1 = result1.stdout + result1.stderr

        if "UNSATISFIABLE" in out1:
            return "NoSolution"

        # --- Sugar 出力 "a <var> <val>" に対応 ---
        assignments = re.findall(r'^a\s+(x_\d+_\d+)\s+([01])$', out1, flags=re.MULTILINE)
        
        if not assignments:
            return "Error: 解のパース失敗"
        
        # --- 否定制約を追加 ---
        with open(csp_filename, "a") as f:
            f.write("\n")
            f.write("(or\n")
            for v, val in assignments:
                f.write(f"  (not (= {v} {val}))\n")
            f.write(")\n")

        # --- 2回目の解をチェック ---
        result2 = subprocess.run(["sugar", csp_filename],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out2 = result2.stdout + result2.stderr
        
        if "UNSATISFIABLE" in out2:
            return "Unique"
        elif "SATISFIABLE" in out2:
            return "Multiple"
        else:
            return "Error: 判定不能"

    except FileNotFoundError:
        return "Error: SugarまたはMiniSATが見つかりません。PATHを確認してください。"
    except Exception as e:
        return f"Error: 実行時エラー ({e})"

# ------------------------- Flask ルート -------------------------
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'ファイルがありません'
        file = request.files['file']
        if file.filename == '':
            return 'ファイルが選択されていません'

        img = process_image(file.stream, size=(30, 30))
        if img is None:
            return '無効な画像ファイルです'

        height, width = img.shape
        grid = [[0 if img[y,x]!=0 else 1 for x in range(width)] for y in range(height)]
        row_hints, col_hints = get_hints(grid)
        csp_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'problem.csp')
        unicity_result = check_unicity(row_hints, col_hints, width, height, csp_filename)

        return render_template('result.html',
                               row_hints=row_hints,
                               col_hints=col_hints,
                               grid=grid,
                               width=width,
                               height=height,
                               unicity=unicity_result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
