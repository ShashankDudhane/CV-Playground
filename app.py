import io
from flask import Flask, request, send_file, render_template, jsonify
import cv2
import numpy as np
from PIL import Image
import base64
import html

app = Flask(__name__, static_folder='static', template_folder='templates')

# ---------- Utility helpers ----------
def read_image_from_bytes(bts):
    # Read uploaded bytes into a BGR OpenCV image
    arr = np.frombuffer(bts, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image")
    # convert RGBA -> BGRA or convert color spaces so we always work in BGR
    if img.ndim == 2:  # grayscale -> convert to BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        # convert BGRA -> BGR (drop alpha) for processing
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def send_mat_as_png(mat):
    # mat is BGR (uint8). encode to PNG and return send_file
    success, encoded = cv2.imencode('.png', mat)
    if not success:
        raise ValueError("Failed to encode image")
    return send_file(io.BytesIO(encoded.tobytes()), mimetype='image/png')

def matrix_to_html(mat, fmt="int"):
    """Return HTML table for a small 2D numpy array"""
    if mat is None:
        return "<i>None</i>"
    try:
        a = np.array(mat)
        if a.ndim != 2:
            # flatten color to grayscale if necessary
            if a.ndim == 3:
                a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
            else:
                a = a.reshape((-1, a.shape[-1]))
        rows = []
        for r in a:
            cells = []
            for v in r:
                if fmt == "float":
                    cells.append(f"<td>{float(v):.4f}</td>")
                else:
                    cells.append(f"<td>{int(v)}</td>")
            rows.append("<tr>" + "".join(cells) + "</tr>")
        return "<table class='table table-sm table-bordered' style='display:inline-block; margin-right:8px;'>" + "".join(rows) + "</table>"
    except Exception as e:
        return f"<pre>Could not render matrix: {html.escape(str(e))}</pre>"

def small_patch(img, size=5):
    """Return a small centered grayscale patch (size x size) as integers"""
    h, w = img.shape[:2]
    cx, cy = w//2, h//2
    half = size//2
    x0, y0 = max(0, cx-half), max(0, cy-half)
    x1, y1 = min(w, cx+half+1), min(h, cy+half+1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    patch = gray[y0:y1, x0:x1]
    return patch

# ---------- Feature / processing functions ----------
def draw_keypoints_on_bgr(img_bgr, keypoints, color=(0,255,0)):
    out = img_bgr.copy()
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(out, (x,y), 3, color, 1, lineType=cv2.LINE_AA)
    return out

def process_image(img, section, technique, params):
    # img: BGR numpy array
    # section: 'intensity', 'smoothing', 'edge', 'corner', 'features'
    # technique: string id
    # params: dict of extra parameters (strings -> cast)
    try:
        if section == 'intensity':
            if technique == 'negation':
                return cv2.bitwise_not(img)
            if technique == 'threshold':
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                thv = int(params.get('th_val', 128))
                thmax = int(params.get('th_max', 255))
                type_s = params.get('th_type', 'binary')
                cvType = cv2.THRESH_BINARY
                if type_s == 'binary_inv': cvType = cv2.THRESH_BINARY_INV
                if type_s == 'trunc': cvType = cv2.THRESH_TRUNC
                if type_s == 'tozero': cvType = cv2.THRESH_TOZERO
                if type_s == 'tozero_inv': cvType = cv2.THRESH_TOZERO_INV
                _, tdst = cv2.threshold(gray, thv, thmax, cvType)
                return cv2.cvtColor(tdst, cv2.COLOR_GRAY2BGR)
            if technique == 'bit_plane':
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                plane = int(params.get('bp_plane', 7))
                bitimg = np.where(((gray >> plane) & 1) == 1, 255, 0).astype(np.uint8)
                return cv2.cvtColor(bitimg, cv2.COLOR_GRAY2BGR)
            if technique == 'gray_slice':
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                low = int(params.get('gs_low', 100))
                high = int(params.get('gs_high', 200))
                preserve = params.get('gs_preserve', 'yes')
                out = np.zeros_like(gray)
                mask = (gray >= low) & (gray <= high)
                out[mask] = 255
                if preserve == 'yes':
                    out[~mask] = gray[~mask]
                return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

        if section == 'smoothing':
            if technique == 'mean':
                k = int(params.get('mean_k', 3))
                return cv2.blur(img, (k,k))
            if technique == 'gaussian':
                k = int(params.get('gaussK', 3))
                return cv2.GaussianBlur(img, (k,k), 0)
            if technique == 'median':
                k = int(params.get('medianK', 3))
                return cv2.medianBlur(img, k)

        if section == 'edge':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if technique == 'sobel':
                k = int(params.get('sobel_k', 3))
                gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=k)
                gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=k)
                abs_gx = cv2.convertScaleAbs(gx)
                abs_gy = cv2.convertScaleAbs(gy)
                grad = cv2.addWeighted(abs_gx, 0.5, abs_gy, 0.5, 0)
                return cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)
            if technique == 'prewitt':
                kx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=np.float32)
                ky = np.array([[-1,-1,-1],[0,0,0],[1,1,1]], dtype=np.float32)
                gx = cv2.filter2D(gray, cv2.CV_32F, kx)
                gy = cv2.filter2D(gray, cv2.CV_32F, ky)
                gx = cv2.convertScaleAbs(gx)
                gy = cv2.convertScaleAbs(gy)
                grad = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)
                return cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)
            if technique == 'roberts':
                kx = np.array([[1,0],[0,-1]], dtype=np.float32)
                ky = np.array([[0,1],[-1,0]], dtype=np.float32)
                gx = cv2.filter2D(gray, cv2.CV_32F, kx)
                gy = cv2.filter2D(gray, cv2.CV_32F, ky)
                gx = cv2.convertScaleAbs(gx)
                gy = cv2.convertScaleAbs(gy)
                grad = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)
                return cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)

        if section == 'corner':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if technique == 'shi':
                maxC = int(params.get('shi_n', 200))
                q = float(params.get('shi_q', 10)) / 100.0
                minD = int(params.get('shi_d', 5))
                corners = cv2.goodFeaturesToTrack(gray, maxC, q, minD)
                out = img.copy()
                if corners is not None:
                    for c in corners:
                        x,y = c.ravel()
                        cv2.circle(out, (int(x),int(y)), 3, (255,0,0), 2)
                return out
            if technique == 'harris':
                block = int(params.get('h_block', 2))
                ksize = int(params.get('h_ksize', 3))
                k = float(params.get('h_k', 4))/100.0
                dst = cv2.cornerHarris(gray, block, ksize, k)
                # apply dilation and threshold fast method
                dst = cv2.dilate(dst, None)
                out = img.copy()
                out[dst > 0.01 * dst.max()] = [0,255,0]
                return out

        if section == 'features':
            # parameters
            maxF = int(params.get('f_nfeatures', 500))
            # Try SIFT, SURF, then ORB
            if technique == 'sift':
                try:
                    sift = cv2.SIFT_create()
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    kps, des = sift.detectAndCompute(gray, None)
                    out = draw_keypoints_on_bgr(img, kps, color=(255,0,0))
                    return out
                except Exception:
                    # fallback ORB
                    orb = cv2.ORB_create(nfeatures=maxF)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    kps = orb.detect(gray, None)
                    out = draw_keypoints_on_bgr(img, kps, color=(0,255,0))
                    return out
            if technique == 'surf':
                try:
                    # SURF requires xfeatures2d (contrib)
                    surf = cv2.xfeatures2d.SURF_create(400)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    kps, des = surf.detectAndCompute(gray, None)
                    out = draw_keypoints_on_bgr(img, kps, color=(0,0,255))
                    return out
                except Exception:
                    orb = cv2.ORB_create(nfeatures=maxF)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    kps = orb.detect(gray, None)
                    out = draw_keypoints_on_bgr(img, kps, color=(0,255,0))
                    return out

        # Default: return copy
        return img.copy()

    except Exception as exc:
        # on any error, return original image and raise
        print("Processing error:", exc)
        return img.copy()

# ---------- Explanation generator ----------
def generate_explanation(img, section, technique, params):
    """
    Build structured HTML explanation steps for the technique.
    We keep matrices small (patches) to make the math readable in the UI.
    """
    steps = []
    try:
        # small sample patch (center)
        patch = small_patch(img, size=5)
        steps.append({
            "title": "Input (grayscale patch center 5×5)",
            "html": matrix_to_html(patch)
        })

        if section == 'edge':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if technique == 'sobel':
                k = int(params.get('sobel_k', 3))
                # show kernels
                kx = cv2.getDerivKernels(1,0,k)[0].reshape(-1,1)  # not simple; we'll show canonical sobel
                # canonical sobel kernels:
                kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
                ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
                steps.append({"title":"Sobel kernel (Gx)","html": matrix_to_html(kx)})
                steps.append({"title":"Sobel kernel (Gy)","html": matrix_to_html(ky)})
                # compute convolution for the patch center (pad if needed)
                pad = cv2.copyMakeBorder(patch,1,1,1,1,cv2.BORDER_REPLICATE)
                # pick central 3x3 inside pad to convolve once (center of patch)
                region = pad[1:4,1:4].astype(np.int32)
                conv_x = int(np.sum(region * kx))
                conv_y = int(np.sum(region * ky))
                steps.append({"title":"Example convolution on central 3×3 region",
                              "html": "<div>Region (3×3): " + matrix_to_html(region) +
                                      "</div><div>Gx = sum(region * Gx) = <b>%d</b></div><div>Gy = sum(region * Gy) = <b>%d</b></div>" % (conv_x, conv_y)})
                # magnitude (approx)
                mag = int(min(255, abs(conv_x)*0.5 + abs(conv_y)*0.5))
                steps.append({"title":"Gradient magnitude (approx)","html": f"<div>|Gx|*0.5 + |Gy|*0.5 = <b>{mag}</b></div>"})
                return steps

            if technique == 'prewitt':
                kx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
                ky = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
                steps.append({"title":"Prewitt kernel (Gx)","html": matrix_to_html(kx)})
                steps.append({"title":"Prewitt kernel (Gy)","html": matrix_to_html(ky)})
                pad = cv2.copyMakeBorder(patch,1,1,1,1,cv2.BORDER_REPLICATE)
                region = pad[1:4,1:4].astype(np.int32)
                conv_x = int(np.sum(region * kx))
                conv_y = int(np.sum(region * ky))
                steps.append({"title":"Example convolution",
                              "html": "<div>Region (3×3): " + matrix_to_html(region) +
                                      "</div><div>Gx = <b>%d</b>, Gy = <b>%d</b></div>" % (conv_x, conv_y)})
                return steps

            if technique == 'roberts':
                kx = np.array([[1,0],[0,-1]])
                ky = np.array([[0,1],[-1,0]])
                steps.append({"title":"Roberts kernels","html": matrix_to_html(kx) + matrix_to_html(ky)})
                pad = cv2.copyMakeBorder(patch,1,1,1,1,cv2.BORDER_REPLICATE)
                region = pad[1:3,1:3].astype(np.int32)
                conv_x = int(np.sum(region * kx))
                conv_y = int(np.sum(region * ky))
                steps.append({"title":"Example convolution",
                              "html": "<div>Region (2×2): " + matrix_to_html(region) +
                                      "</div><div>Gx = <b>%d</b>, Gy = <b>%d</b></div>" % (conv_x, conv_y)})
                return steps

        if section == 'intensity':
            if technique == 'threshold':
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                thv = int(params.get('th_val', 128))
                thmax = int(params.get('th_max', 255))
                type_s = params.get('th_type', 'binary')
                steps.append({"title":"Threshold parameters",
                              "html": f"<div>Threshold value = <b>{thv}</b>, Max = <b>{thmax}</b>, Type = <b>{html.escape(type_s)}</b></div>"})
                # show how threshold applied on the patch
                patch_vals = patch.astype(np.int32)
                out_patch = np.where(patch_vals > thv, thmax, 0).astype(np.int32) if type_s == 'binary' else None
                if out_patch is not None:
                    steps.append({"title":"Patch after threshold (binary)","html": matrix_to_html(out_patch)})
                else:
                    steps.append({"title":"Patch after threshold","html":"<div>See original image patch; this threshold type modifies pixels accordingly.</div>"})
                return steps

            if technique == 'bit_plane':
                plane = int(params.get('bp_plane', 7))
                steps.append({"title":"Bit-plane parameters", "html": f"<div>Selected plane: <b>{plane}</b></div>"})
                # show binary of central patch values
                patch_bin = np.vectorize(lambda x: format(int(x),'08b'))(patch)
                # show central pixel binary and extraction
                cent = patch[patch.shape[0]//2, patch.shape[1]//2]
                cent_bin = format(int(cent),'08b')
                bit_val = (cent >> plane) & 1
                steps.append({"title":"Example central pixel", "html": f"<div>Central value = <b>{int(cent)}</b> (binary {cent_bin}) → bit at plane {plane} = <b>{bit_val}</b></div>"})
                return steps

            if technique == 'gray_slice':
                low = int(params.get('gs_low', 100))
                high = int(params.get('gs_high', 200))
                preserve = params.get('gs_preserve', 'yes')
                steps.append({"title":"Gray-level slicing params", "html": f"<div>Low=<b>{low}</b>, High=<b>{high}</b>, Preserve outside=<b>{preserve}</b></div>"})
                mask = (patch >= low) & (patch <= high)
                steps.append({"title":"Patch mask (1=inside slice)", "html": matrix_to_html(mask.astype(np.int32))})
                return steps

        if section == 'smoothing':
            if technique == 'mean':
                k = int(params.get('mean_k', 3))
                kernel = np.ones((k,k), dtype=np.float32) / (k*k)
                steps.append({"title":"Mean filter kernel", "html": matrix_to_html((kernel*255).astype(np.int32))})
                # show convolution on central region
                pad = cv2.copyMakeBorder(patch, k//2, k//2, k//2, k//2, cv2.BORDER_REPLICATE)
                region = pad[0:k,0:k].astype(np.float32)
                conv_val = float(np.sum(region * kernel))
                steps.append({"title":"Example convolution value (first k×k region)","html": f"<div>Region: {matrix_to_html(region)}<div>Convolved value (avg) ≈ <b>{conv_val:.2f}</b></div></div>"})
                return steps

            if technique == 'gaussian':
                k = int(params.get('gaussK', 3))
                # build gaussian kernel
                gkern_1d = cv2.getGaussianKernel(k, -1)
                gk = gkern_1d @ gkern_1d.T
                steps.append({"title":"Gaussian kernel (normalized)", "html": matrix_to_html((gk*1000).astype(np.int32))})
                pad = cv2.copyMakeBorder(patch, k//2, k//2, k//2, k//2, cv2.BORDER_REPLICATE)
                region = pad[0:k,0:k].astype(np.float32)
                conv_val = float(np.sum(region * gk))
                steps.append({"title":"Example convolution value","html": f"<div>Convolved value ≈ <b>{conv_val:.2f}</b></div>"})
                return steps

            if technique == 'median':
                k = int(params.get('medianK', 3))
                steps.append({"title":"Median filter", "html": f"<div>Kernel size = <b>{k}</b>. Median filter picks median of neighborhood.</div>"})
                med_val = np.median(patch[0:k,0:k])
                steps.append({"title":"Example median on first k×k region", "html": f"<div>Median ≈ <b>{int(med_val)}</b></div>"})
                return steps

        if section == 'corner':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if technique == 'shi':
                maxC = int(params.get('shi_n', 200))
                q = float(params.get('shi_q', 10)) / 100.0
                minD = int(params.get('shi_d', 5))
                corners = cv2.goodFeaturesToTrack(gray, maxCorners=maxC, qualityLevel=q, minDistance=minD)
                count = 0 if corners is None else len(corners)
                steps.append({"title":"Shi–Tomasi parameters", "html": f"<div>maxCorners=<b>{maxC}</b>, quality=<b>{q:.3f}</b>, minDistance=<b>{minD}</b></div>"})
                steps.append({"title":"Detected corners (sample)", "html": f"<div>Count: <b>{count}</b></div>"})
                return steps

            if technique == 'harris':
                block = int(params.get('h_block', 2))
                ksize = int(params.get('h_ksize', 3))
                k = float(params.get('h_k', 4))/100.0
                dst = cv2.cornerHarris(gray, block, ksize, k)
                norm = cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX)
                small = cv2.resize(norm, (5,5))
                steps.append({"title":"Harris response (normalized small)","html": matrix_to_html(small, fmt="float")})
                steps.append({"title":"Harris params","html": f"<div>blockSize={block}, ksize={ksize}, k={k:.3f}</div>"})
                return steps

        if section == 'features':
            maxF = int(params.get('f_nfeatures', 500))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if technique == 'sift':
                try:
                    sift = cv2.SIFT_create()
                    kps, des = sift.detectAndCompute(gray, None)
                    n = 0 if kps is None else len(kps)
                    steps.append({"title":"SIFT explanation","html": f"<div>Detected keypoints: <b>{n}</b> (showing locations not printed here). SIFT computes scale-space extrema and descriptors per keypoint.</div>"})
                    return steps
                except Exception:
                    orb = cv2.ORB_create(nfeatures=maxF)
                    kps = orb.detect(gray, None)
                    n = 0 if kps is None else len(kps)
                    steps.append({"title":"SIFT not available — ORB fallback","html": f"<div>ORB detected <b>{n}</b> keypoints.</div>"})
                    return steps
            if technique == 'surf':
                try:
                    surf = cv2.xfeatures2d.SURF_create(400)
                    kps, des = surf.detectAndCompute(gray, None)
                    n = 0 if kps is None else len(kps)
                    steps.append({"title":"SURF explanation","html": f"<div>Detected keypoints: <b>{n}</b></div>"})
                    return steps
                except Exception:
                    orb = cv2.ORB_create(nfeatures=maxF)
                    kps = orb.detect(gray, None)
                    n = 0 if kps is None else len(kps)
                    steps.append({"title":"SURF not available — ORB fallback","html": f"<div>ORB detected <b>{n}</b> keypoints.</div>"})
                    return steps

        # Fallback message
        steps.append({"title":"Note","html":"<div>No detailed calculation available for this technique (yet).</div>"})
        return steps

    except Exception as e:
        return [{"title":"Error","html": f"<pre>{html.escape(str(e))}</pre>"}]


# ---------- Flask routes ----------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    """
    Handles single-image processing (upload from file input)
    Expects:
      - 'image' file (multipart)
      - 'section' (string)
      - 'technique' (string)
      - any extra params in form fields
    Returns PNG image
    """
    if 'image' not in request.files:
        return "No image file", 400
    file = request.files['image']
    img_bytes = file.read()
    img = read_image_from_bytes(img_bytes)

    section = request.form.get('section', 'intensity')
    technique = request.form.get('technique', '')
    # gather params
    params = {k: request.form.get(k) for k in request.form.keys()}
    # process
    out = process_image(img, section, technique, params)
    return send_mat_as_png(out)

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """
    Handles webcam frames (sent as raw image bytes in body or as multipart 'frame')
    Accepts content-type multipart/form-data with 'frame' or raw octet-stream.
    Also accepts form fields: section, technique, and params.
    """
    bts = None
    if 'frame' in request.files:
        bts = request.files['frame'].read()
    else:
        bts = request.data
    if not bts:
        return "No frame data", 400
    img = read_image_from_bytes(bts)
    section = request.form.get('section', 'features')
    technique = request.form.get('technique', 'sift')
    params = {k: request.form.get(k) for k in request.form.keys()}
    out = process_image(img, section, technique, params)
    return send_mat_as_png(out)

@app.route('/explain', methods=['POST'])
def explain():
    """
    New endpoint: returns JSON with explanation steps (HTML fragments)
    Accepts same form fields (image/frame, section, technique, params)
    """
    bts = None
    if 'image' in request.files:
        bts = request.files['image'].read()
    elif 'frame' in request.files:
        bts = request.files['frame'].read()
    else:
        bts = request.data
    if not bts:
        return jsonify({"error":"No image/frame provided"}), 400
    try:
        img = read_image_from_bytes(bts)
    except Exception as e:
        return jsonify({"error": f"Could not decode image: {str(e)}"}), 400

    section = request.form.get('section', 'intensity')
    technique = request.form.get('technique', '')
    params = {k: request.form.get(k) for k in request.form.keys()}

    steps = generate_explanation(img, section, technique, params)
    # steps is a list of dicts {title, html}
    return jsonify({"technique": technique, "section": section, "steps": steps})

if __name__ == '__main__':
    app.run(debug=True)
