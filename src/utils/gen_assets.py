import os
from PIL import Image, ImageDraw

def ensure_assets():
    assets_dir = os.path.join(os.path.dirname(__file__), '..', 'assets')
    os.makedirs(assets_dir, exist_ok=True)

    # Background
    bg = Image.new('RGBA', (1024, 1024), (235, 255, 235, 255))
    draw = ImageDraw.Draw(bg)
    for x in range(0, 1024, 64):
        draw.line([(x, 0), (x, 1024)], fill=(200, 230, 200, 255), width=1)
    for y in range(0, 1024, 64):
        draw.line([(0, y), (1024, y)], fill=(200, 230, 200, 255), width=1)
    bg.save(os.path.join(assets_dir, 'background.png'))

    # UAV eagle
    uav = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
    d = ImageDraw.Draw(uav)
    d.ellipse([32, 32, 224, 224], fill=(30, 144, 255, 255), outline=(255, 255, 255, 255), width=4)
    d.polygon([(128, 40), (160, 100), (96, 100)], fill=(255, 255, 255, 255))
    uav.save(os.path.join(assets_dir, 'eagle.png'))

    # Chick
    chick = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
    d = ImageDraw.Draw(chick)
    d.ellipse([40, 40, 216, 216], fill=(255, 215, 0, 255), outline=(255, 165, 0, 255), width=4)
    chick.save(os.path.join(assets_dir, 'chick.png'))

    # Protector 
    protector = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
    d = ImageDraw.Draw(protector)
    d.polygon([(128, 32), (224, 224), (32, 224)], fill=(220, 20, 60, 255), outline=(139, 0, 0, 255))
    protector.save(os.path.join(assets_dir, 'protector.png'))

if __name__ == '__main__':
    ensure_assets()
    print('Assets generated in src/assets')