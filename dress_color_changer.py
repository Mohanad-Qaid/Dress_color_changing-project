import cv2
import numpy as np
import argparse

# Global variables
original_image = None
hls_image = None
target_hls = None


def rgb_to_hls(bgr_img):
    """Convert BGR to HLS: H in [0,360], L in [0,1], S in [0,1]"""
    img = bgr_img.astype(np.float64) / 255.0
    B, G, R = img[:,:,0], img[:,:,1], img[:,:,2]
    
    Cmax = np.maximum(np.maximum(R, G), B)
    Cmin = np.minimum(np.minimum(R, G), B)
    Delta = Cmax - Cmin

    # Lightness: L = (Cmax + Cmin) / 2
    L = (Cmax + Cmin) / 2.0

    # Saturation: S = Delta / (1 - |2L - 1|)
    S = np.zeros_like(L)
    mask = Delta > 1e-10
    denom = np.maximum(1.0 - np.abs(2.0 * L - 1.0), 1e-10)
    S[mask] = Delta[mask] / denom[mask]
    S = np.clip(S, 0, 1)

    # Hue calculation based on which channel is max
    H = np.zeros_like(L)
    
    mask_r = (Cmax == R) & mask
    H[mask_r] = 60.0 * (((G[mask_r] - B[mask_r]) / Delta[mask_r]) % 6)
    
    mask_g = (Cmax == G) & mask & ~mask_r
    H[mask_g] = 60.0 * ((B[mask_g] - R[mask_g]) / Delta[mask_g] + 2)
    
    mask_b = (Cmax == B) & mask & ~mask_r & ~mask_g
    H[mask_b] = 60.0 * ((R[mask_b] - G[mask_b]) / Delta[mask_b] + 4)
    
    H = H % 360
    
    return np.stack([H, L, S], axis=2)


def hls_to_rgb(hls_img):
    """Convert HLS back to BGR"""
    H, L, S = hls_img[:,:,0], hls_img[:,:,1], hls_img[:,:,2]
    
    C = (1.0 - np.abs(2.0 * L - 1.0)) * S
    
    H_prime = H / 60.0
    X = C * (1.0 - np.abs(H_prime % 2 - 1.0))
    
    R, G, B = np.zeros_like(H), np.zeros_like(H), np.zeros_like(H)
    
    m0 = (H_prime >= 0) & (H_prime < 1)
    m1 = (H_prime >= 1) & (H_prime < 2)
    m2 = (H_prime >= 2) & (H_prime < 3)
    m3 = (H_prime >= 3) & (H_prime < 4)
    m4 = (H_prime >= 4) & (H_prime < 5)
    m5 = (H_prime >= 5) & (H_prime < 6)
    
    R[m0], G[m0], B[m0] = C[m0], X[m0], 0
    R[m1], G[m1], B[m1] = X[m1], C[m1], 0
    R[m2], G[m2], B[m2] = 0, C[m2], X[m2]
    R[m3], G[m3], B[m3] = 0, X[m3], C[m3]
    R[m4], G[m4], B[m4] = X[m4], 0, C[m4]
    R[m5], G[m5], B[m5] = C[m5], 0, X[m5]
    
    m = L - C / 2.0
    
    bgr = np.stack([(B + m) * 255, (G + m) * 255, (R + m) * 255], axis=2)
    return np.clip(bgr, 0, 255).astype(np.uint8)


def hue_distance(h1, h2):
    """Circular distance for hue values"""
    diff = np.abs(h1 - h2)
    return np.minimum(diff, 360.0 - diff)


def on_mouse_click(event, x, y, flags, param):
    global target_hls, hls_image
    if event == cv2.EVENT_LBUTTONDOWN and hls_image is not None:
        target_hls = hls_image[y, x].copy()
        print(f"Selected: H={target_hls[0]:.1f}, L={target_hls[1]:.3f}, S={target_hls[2]:.3f}")
        update_display()


def update_display(*args):
    global original_image, hls_image, target_hls
    
    if original_image is None:
        return
    if target_hls is None:
        cv2.imshow('Color Changer', original_image)
        return
    
    # Get trackbar values
    new_h = cv2.getTrackbarPos('New Hue', 'Controls') * 2.0
    new_s = cv2.getTrackbarPos('New Sat', 'Controls') / 255.0
    h_tol = cv2.getTrackbarPos('Hue Tol', 'Controls')
    l_tol = cv2.getTrackbarPos('Light Tol', 'Controls') / 255.0
    s_tol = cv2.getTrackbarPos('Sat Tol', 'Controls') / 255.0
    show_mask = cv2.getTrackbarPos('Show Mask', 'Controls')
    
    # Creating mask based on color distance
    h_dist = hue_distance(hls_image[:,:,0], target_hls[0])
    l_dist = np.abs(hls_image[:,:,1] - target_hls[1])
    s_dist = np.abs(hls_image[:,:,2] - target_hls[2])
    
    mask = (h_dist <= h_tol) & (l_dist <= l_tol) & (s_dist <= s_tol)
    
    if show_mask == 1:
        result = original_image.copy()
        result[mask] = [0, 255, 0]
    else:
        modified_hls = hls_image.copy()
        modified_hls[mask, 0] = new_h
        modified_hls[mask, 2] = new_s
        result = hls_to_rgb(modified_hls)
    
    cv2.imshow('Color Changer', result)


def main():
    global original_image, hls_image
    
    parser = argparse.ArgumentParser(description="Color Changer")
    parser.add_argument("-i", "--image", required=True, help="Image path")
    args = parser.parse_args()
    
    original_image = cv2.imread(args.image)
    if original_image is None:
        print(f"Error: Cannot load {args.image}")
        return
    
    h, w = original_image.shape[:2]
    if h > 700:
        scale = 700 / h
        original_image = cv2.resize(original_image, (int(w * scale), int(h * scale)))
    
    # Convert to HLS
    hls_image = rgb_to_hls(original_image)
    
    # Setup UI
    cv2.namedWindow('Color Changer')
    cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Controls', 500, 300)
    cv2.setMouseCallback('Color Changer', on_mouse_click)
    
    cv2.createTrackbar('New Hue', 'Controls', 0, 180, update_display)
    cv2.createTrackbar('New Sat', 'Controls', 200, 255, update_display)
    cv2.createTrackbar('Hue Tol', 'Controls', 15, 90, update_display)
    cv2.createTrackbar('Light Tol', 'Controls', 60, 128, update_display)
    cv2.createTrackbar('Sat Tol', 'Controls', 60, 128, update_display)
    cv2.createTrackbar('Show Mask', 'Controls', 0, 1, update_display)
    
    print("Click on image to select color. Press 'q' to quit.")
    update_display()
    
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()

