"""Conservative screenshot-only sun candidate locator for bounded day04 tests.

This is a colour/shape heuristic, not general sun/plant/occupancy perception.
Results require manual post-action confirmation and a reviewed search region.
"""
from PIL import Image


def sun_candidates(image_path, box):
    left, top, right, bottom = box
    if any(type(v) is not int for v in box) or not 40 <= left < right <= 760 or not 95 <= top < bottom <= 590:
        raise ValueError('Sun search box must be an integer rectangle inside the board')
    with Image.open(image_path) as image:
        crop = image.convert('RGB').crop(box)
    pixels = crop.load()
    bright = {(x,y) for y in range(crop.height) for x in range(crop.width)
              if (lambda p: p[0] >= 235 and p[1] >= 230 and p[2] <= 160 and p[0]-p[1] <= 20)(pixels[x,y])}
    result = []
    while bright:
        seed = bright.pop()
        component, stack = [seed], [seed]
        while stack:
            x,y = stack.pop()
            for point in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                if point in bright:
                    bright.remove(point)
                    component.append(point)
                    stack.append(point)
        if len(component) < 70:
            continue
        xs,ys = zip(*component)
        width,height = max(xs)-min(xs)+1,max(ys)-min(ys)+1
        if not 9 <= width <= 42 or not 9 <= height <= 42 or not .65 <= width/height <= 1.5:
            continue
        x,y = round((min(xs)+max(xs))/2),round((min(ys)+max(ys))/2)
        core = {(cx,cy) for cy in range(y-3,y+4) for cx in range(x-3,x+4)}
        if len(core.intersection(component)) < 40:
            continue
        result.append({'point_native':[left+x,top+y], 'core_pixels':len(component),
                       'method':'experimental_bright_disk_in_reviewed_region'})
    return sorted(result,key=lambda item:(item['point_native'][1],item['point_native'][0]))
