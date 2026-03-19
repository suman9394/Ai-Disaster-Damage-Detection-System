from exif import Image as ExifImage

def get_coordinates(image_bytes):
    img = ExifImage(image_bytes)
    if img.has_exif:
        try:
            def to_decimal(coords, ref):
                decimal = coords[0] + coords[1] / 60 + coords[2] / 3600
                return -decimal if ref in ['S', 'W'] else decimal

            lat = to_decimal(img.gps_latitude, img.gps_latitude_ref)
            lon = to_decimal(img.gps_longitude, img.gps_longitude_ref)
            return lat, lon
        except:
            return None, None
    return None, None