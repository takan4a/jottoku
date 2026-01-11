import os
from flask import Flask, render_template, request
import cv2
import numpy as np
from PIL import Image
import io
import subprocess
import re
import tempfile

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ------------------------- 画像処理関数 -------------------------
def process_image(image_stream, size=(30, 30)):
    input_bytes = image_stream.read()
    # rembg は重い依存があるため遅延インポートする（テスト時の副作用軽減）
    try:
        from rembg import remove
        out_bytes = remove(input_bytes)
    except Exception:
        # rembg が利用できない場合は入力をそのまま使う
        out_bytes = input_bytes
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
        # 保存しておく（adapt_puzzle / solve で利用）
        global LATEST_ROW_HINTS, LATEST_COL_HINTS, LATEST_WIDTH, LATEST_HEIGHT
        LATEST_ROW_HINTS = row_hints
        LATEST_COL_HINTS = col_hints
        LATEST_HEIGHT = height
        LATEST_WIDTH = width
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

# ------------------------- パズル適応アルゴリズム -------------------------
def run_sugar_with_constraints(row_hints, col_hints, width, height, extra_constraints, csp_path=None):
    """補助: 一時CSPファイルを作り、追加制約を付けて sugar を実行する。"""
    if csp_path is None:
        tf = tempfile.NamedTemporaryFile(mode='w', suffix='.csp', delete=False, dir=app.config['UPLOAD_FOLDER'])
        csp_path = tf.name
        tf.close()
    # ベースCSPを書き出す
    if not generate_csp_file(row_hints, col_hints, width, height, csp_path):
        return None, "Error: CSP生成失敗"

    # 追加制約を追記
    if extra_constraints:
        with open(csp_path, 'a') as f:
            f.write('\n')
            for cons in extra_constraints:
                f.write(cons + '\n')

    try:
        os.environ["PATH"] += ":/home/takana/projects/jottoku/sugar-2.3.4/bin"
        result = subprocess.run(["sugar", csp_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout + result.stderr, None
    except FileNotFoundError:
        return None, "Error: SugarまたはMiniSATが見つかりません。PATHを確認してください。"
    except Exception as e:
        return None, f"Error: 実行時エラー ({e})"


def solve(current_image):
    """簡易ソルバー: グローバルに保存された最新ヒントを用い、各マスが強制されるかを判定する。
    返り値は同形の numpy 配列で、確定値は 0/1、未確定は -1 を返す。
    注意: 小さなパズル向けの単純実装（各マスごとに2回SATチェック）。"""
    try:
        row_hints = LATEST_ROW_HINTS
        col_hints = LATEST_COL_HINTS
        height = LATEST_HEIGHT
        width = LATEST_WIDTH
    except NameError:
        # ヒントが未保存の場合は、現在のグリッドをそのまま返す（未確定扱い）
        arr = np.full_like(current_image, -1)
        arr[current_image == 1] = 1
        arr[current_image == 0] = -1
        return arr

    solution = np.full((height, width), -1, dtype=int)

    # 既に確定しているセルはそのまま反映
    solution[current_image == 1] = 1
    solution[current_image == 0] = -1

    # 各未確定セルについて、0/1 を強制したときにSATかを確認
    for y in range(height):
        for x in range(width):
            if current_image[y][x] == 1:
                solution[y][x] = 1
                continue
            # 既に黒にして試行中のセルも -1 として扱う
            # 0 を強制してSATか確認
            cons0 = [f"(= x_{y}_{x} 0)"]
            out0, err0 = run_sugar_with_constraints(row_hints, col_hints, width, height, cons0)
            sat0 = False
            if out0 is not None and "UNSATISFIABLE" not in out0:
                sat0 = True

            # 1 を強制してSATか確認
            cons1 = [f"(= x_{y}_{x} 1)"]
            out1, err1 = run_sugar_with_constraints(row_hints, col_hints, width, height, cons1)
            sat1 = False
            if out1 is not None and "UNSATISFIABLE" not in out1:
                sat1 = True

            if sat0 and not sat1:
                solution[y][x] = 0
            elif sat1 and not sat0:
                solution[y][x] = 1
            else:
                solution[y][x] = -1

    return solution


def adapt_puzzle(current_image, original_gray_image, alpha=8.0, beta=1.0):
    """
    Algorithm 5: AdaptPuzzle の実装
    current_image: 現在の白黒パズル (0:白, 1:黒)
    original_gray_image: 元のグレースケール画像 (0:黒 〜 255:白)
    """

    current_solution = solve(current_image.copy())

    # すでに全てのマスが確定していれば終了
    if count_unknowns(current_solution) == 0:
        return current_image

    best_cell = None
    min_value = float('inf')

    height, width = current_image.shape

    for y in range(height):
        for x in range(width):
            # 条件: 現在白マスかつソルバーが未確定
            if current_image[y][x] == 0 and current_solution[y][x] == -1:
                # 一時的に黒にして試す
                current_image[y][x] = 1
                simulated_solution = solve(current_image.copy())
                num_unknowns = count_unknowns(simulated_solution)

                pixel_intensity = int(original_gray_image[y][x])

                value = (alpha * num_unknowns) + (beta * pixel_intensity)

                if value < min_value:
                    min_value = value
                    best_cell = (y, x)

                # 元に戻す
                current_image[y][x] = 0

    if best_cell:
        best_y, best_x = best_cell
        current_image[best_y][best_x] = 1
        print(f"Selected cell ({best_x}, {best_y}) with score {min_value}")
    else:
        print("変更可能な候補がありません（エラーまたは完了）")

    return current_image


def count_unknowns(solution):
    return int((solution == -1).sum())
