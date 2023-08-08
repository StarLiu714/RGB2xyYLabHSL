import pandas as pd
import math

def rgb_to_hsl(r, g, b):
    r /= 255.0
    g /= 255.0
    b /= 255.0
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    delta = max_val - min_val
    l = (max_val + min_val) / 2
    if delta == 0:
        h = 0
        s = 0
    else:
        s = delta / (2 - max_val - min_val) if l > 0.5 else delta / (max_val + min_val)
        if max_val == r:
            h = (g - b) / delta + (6 if g < b else 0)
        elif max_val == g:
            h = (b - r) / delta + 2
        else:
            h = (r - g) / delta + 4
        h /= 6
    return h * 360, s * 100, l * 100

def rgb_to_xyz(r, g, b):
    """Converts RGB to XYZ colorspace."""
    # Normalize, apply gamma correction, and scale RGB values
    r = ((r / 255.0) / 12.92 if r <= 10.13725 else ((r / 255.0 + 0.055) / 1.055) ** 2.4) * 100.0
    g = ((g / 255.0) / 12.92 if g <= 10.13725 else ((g / 255.0 + 0.055) / 1.055) ** 2.4) * 100.0
    b = ((b / 255.0) / 12.92 if b <= 10.13725 else ((b / 255.0 + 0.055) / 1.055) ** 2.4) * 100.0
    # Convert RGB to XYZ using the official conversion matrix
    X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    return X, Y, Z

def rgb_to_xyy(r, g, b):
    """Converts RGB to xyY colorspace."""
    X, Y, Z = rgb_to_xyz(r, g, b)
    denom = X + Y + Z
    if denom == 0:
        return 0, 0, Y
    x = X / denom
    y = Y / denom
    return x, y, Y

def f(t):
    """Helper function for Lab conversion."""
    if t > 0.008856:
        return t ** (1/3)
    else:
        return 7.787 * t + 16/116

def rgb_to_lab(r, g, b):
    """Converts RGB to Lab colorspace."""
    # Convert RGB to XYZ
    X, Y, Z = rgb_to_xyz(r, g, b)
    # D65 illuminant normalization constants
    Xn, Yn, Zn = 95.047, 100.000, 108.883
    # Normalize the XYZ values
    X /= Xn
    Y /= Yn
    Z /= Zn
    # Conversion to Lab using the conversion formula
    fx = f(X)
    fy = f(Y)
    fz = f(Z)
    L = max(0, 116 * fy - 16)
    a = (fx - fy) * 500
    b = (fy - fz) * 200
    return L, a, b

def deltaE76(lab1, lab2):
    """Calculate the color difference using Delta E76 formula."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    return math.sqrt((L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)

# Implementing Delta E00
def deltaE00(lab1, lab2, K_L=1, K_C=1, K_H=1):
    """Calculate the color difference using Delta E00 formula."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    L_ = (L1 + L2) / 2.0
    C_ = (C1 + C2) / 2.0
    C1_4 = C1**4
    C2_4 = C2**4
    F = math.sqrt(C1_4 / (C1_4 + 1900.0))
    a1_prime = a1 + a1 / 2.0 * (1 - F)
    a2_prime = a2 + a2 / 2.0 * (1 - F)
    C1_prime = math.sqrt(a1_prime**2 + b1**2)
    C2_prime = math.sqrt(a2_prime**2 + b2**2)
    C__prime = (C1_prime + C2_prime) / 2.0
    h1_rad = math.atan2(b1, a1_prime)
    h2_rad = math.atan2(b2, a2_prime)
    if h1_rad < 0:
        h1_rad += 2*math.pi
    if h2_rad < 0:
        h2_rad += 2*math.pi
    if abs(h1_rad - h2_rad) > math.pi:
        if h2_rad <= h1_rad:
            h2_rad += 2*math.pi
        else:
            h1_rad += 2*math.pi
    H_ = (h1_rad + h2_rad) / 2.0
    T = 1 - 0.17 * math.cos(H_ - math.pi/6) + 0.24 * math.cos(2*H_) + 0.32 * math.cos(3*H_ + math.pi/30) - 0.2 * math.cos(4*H_ - 21*math.pi/60)
    h_ = h2_rad - h1_rad
    if abs(h_) > math.pi:
        if h2_rad <= h1_rad:
            h_ += 2*math.pi
        else:
            h_ -= 2*math.pi
    L__ = (L1 - L2)
    C__ = C1_prime - C2_prime
    H__ = h_ * K_H * T
    S_L = 1 + (0.015 * (L_ - 50)**2) / math.sqrt(20 + (L_ - 50)**2)
    S_C = 1 + 0.045 * C__
    S_H = 1 + 0.015 * C__ * T
    R_T = -2 * math.sqrt(C__prime**7 / (C__prime**7 + 25**7)) * math.sin(60 * math.exp(-((H_ - 275) / 25)**2))
    deltaE = math.sqrt((L__ / (K_L * S_L))**2 + (C__ / (K_C * S_C))**2 + (H__ / (K_H * S_H))**2 + R_T * (C__ / (K_C * S_C)) * (H__ / (K_H * S_H)))
    return deltaE


def compute_colorspaces_from_csv(input_csv, output_csv, colorspaces="all", deltaE_type="76"):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(input_csv)
    
    # Convert RGB columns to numeric format
    df['R'] = pd.to_numeric(df['R'], errors='coerce')
    df['G'] = pd.to_numeric(df['G'], errors='coerce')
    df['B'] = pd.to_numeric(df['B'], errors='coerce')
    
    # Compute HSL values if specified
    if colorspaces in ["HSL", "all"]:
        df['H%'], df['S%'], df['L%'] = zip(*df.apply(lambda row: rgb_to_hsl(row['R'], row['G'], row['B']), axis=1))
    
    # Compute xyY values if specified
    if colorspaces in ["xyY", "all"]:
        df['x'], df['y'], df['Y'] = zip(*df.apply(lambda row: rgb_to_xyy(row['R'], row['G'], row['B']), axis=1))
    
    # Compute Lab values if specified
    if colorspaces in ["Lab", "all"]:
        df['L*'], df['a*'], df['b*'] = zip(*df.apply(lambda row: rgb_to_lab(row['R'], row['G'], row['B']), axis=1))

    # If Lab values are available, compute deltaE values
    if 'L*' in df.columns and 'a*' in df.columns and 'b*' in df.columns:
        lce_rows = df[df.iloc[:, 0] == 'LCE']
        for idx, row in df.iterrows():
            if idx == 0 or len(lce_rows[lce_rows.index < idx]) == 0 or row.iloc[0] == 'LCE' or row.iloc[0] == 'Glass':
                continue
            lce_row = lce_rows[lce_rows.index < idx].iloc[-1]
            lab1 = (row['L*'], row['a*'], row['b*'])
            lab2 = (lce_row['L*'], lce_row['a*'], lce_row['b*'])
            if deltaE_type == "76":
                df.at[idx, 'deltaE76'] = deltaE76(lab1, lab2)
            elif deltaE_type == "00":
                df.at[idx, 'deltaE00'] = deltaE00(lab1, lab2)
    
    # Save the updated DataFrame to the output CSV file
    df.to_csv(output_csv, index=False)

