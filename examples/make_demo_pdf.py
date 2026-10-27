"""Generate an original Persian/English demonstration PDF with a raster table."""
import argparse
from io import BytesIO
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont


def make_demo_pdf(output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_htmlbox(fitz.Rect(45, 45, 550, 720), """
        <div dir="rtl" style="font-size:18px;line-height:1.7">
        <h1>نمونه آموزشی پرسش و پاسخ فارسی</h1>
        <p>این سند برای نمایش نرم‌افزار ساخته شده است. همه نام روش‌ها و اعداد آن ساختگی هستند و نتیجه پژوهش واقعی نیستند.</p>
        <h2>مقایسه دو روش آزمایشی</h2>
        <p>در آزمایش ساختگی، مقدار AUC برای روش آلفا برابر 0.91 و برای روش بتا برابر 0.84 است. روش آلفا مقدار AUC بیشتری دارد.</p>
        <p>زمان پردازش هر سند برای روش آلفا 12 ثانیه و برای روش بتا 8 ثانیه در نظر گرفته شده است. روش بتا سریع‌تر است.</p>
        <p>هدف نمایش، بازیابی شواهد و ارجاع به صفحه و پاراگراف است. این سند اطلاعاتی درباره قیمت سخت‌افزار یا هزینه آموزش ارائه نمی‌کند.</p>
        </div>
        <p style="font-size:14px">Synthetic example: Alpha AUC = 0.91; Beta AUC = 0.84.
        Alpha processing time = 12 seconds; Beta processing time = 8 seconds. These are invented demonstration values.</p>
    """)
    page = doc.new_page()
    page.insert_htmlbox(fitz.Rect(45, 40, 550, 140), '<h1 dir="rtl">جدول نتایج ساختگی</h1><p dir="rtl">برای مقایسه مقدار AUC و زمان پردازش به جدول زیر مراجعه کنید.</p>')
    image = Image.new("RGB", (1100, 420), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=36)
    rows = [("Method", "AUC", "Time (seconds)"), ("Alpha", "0.91", "12"), ("Beta", "0.84", "8")]
    for row_index, values in enumerate(rows):
        y = 30 + row_index * 120
        draw.rectangle((20, y, 1080, y + 115), fill="#e8eef7" if row_index == 0 else "#f7f9fc", outline="#334155", width=2)
        for x, value in zip((45, 430, 690), values):
            draw.text((x, y + 35), value, font=font, fill="#14233c")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    page.insert_image(fitz.Rect(45, 180, 550, 375), stream=buffer.getvalue())
    page.insert_htmlbox(fitz.Rect(45, 430, 550, 600), '<p dir="rtl" style="font-size:18px">محدودیت: این داده‌ها کوچک و ساختگی هستند. از این مثال نمی‌توان درباره کیفیت سامانه روی مقاله‌های واقعی نتیجه‌گیری کرد.</p>')
    doc.set_metadata({"title": "Original synthetic Persian QA demonstration", "author": "Persian Document Understanding project"})
    doc.save(output)
    doc.close()
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("app_data/demo.pdf"))
    print(make_demo_pdf(parser.parse_args().output))
