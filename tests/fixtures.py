"""Clearly synthetic declarations. No real project or company data."""
from io import BytesIO
from pathlib import Path
import fitz
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas


def font_paths():
    regular = [Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),Path('/System/Library/Fonts/Supplemental/Arial.ttf'),Path('C:/Windows/Fonts/arial.ttf')]
    bold = [Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'),Path('/System/Library/Fonts/Supplemental/Arial Bold.ttf'),Path('C:/Windows/Fonts/arialbd.ttf')]
    return next(p for p in regular if p.exists()), next(p for p in bold if p.exists())


def declaration(scan=False, date='01.09.2026', second=True):
    normal, bold = font_paths()
    pdfmetrics.registerFont(TTFont('Test', str(normal)))
    pdfmetrics.registerFont(TTFont('TestBold', str(bold)))
    buffer=BytesIO(); c=Canvas(buffer,pagesize=(595,842)); y=0
    def page():
        nonlocal y
        y=800
        c.setFont('Test',10);c.drawString(28,y,'УЧЕБНЫЙ ПРИМЕР. ВЫМЫШЛЕННЫЕ ДАННЫЕ');y-=25
    def text(t):
        nonlocal y
        c.setFont('Test',12);c.drawString(28,y,t);y-=25
    def field(code,label,value):
        nonlocal y
        top=y+14; bottom=y-33
        c.setLineWidth(.5)
        c.rect(145,bottom,410,47)
        c.line(210,bottom,210,top)
        c.setFont('Test',10);c.drawString(153,y,code)
        c.setFont('Test',10);c.drawString(218,y,label+':')
        c.setFont('TestBold',11);c.drawString(218,y-19,value)
        y-=47
    page();text('ПРОЕКТНАЯ ДЕКЛАРАЦИЯ');text(f'№ 77-009999 от {date}')
    text('Учебный жилой комплекс «Сад»');text('Дата первичного размещения: 01.01.2026')
    field('1.1.3','Краткое наименование','ООО «Учебный застройщик»')
    field('2.1.1','ИНН','7700000000')
    for no in range(1,3 if second else 2):
        if y<400:c.showPage();page()
        field('9.2.1','Вид объекта','Многоквартирный дом')
        field('9.2.2','Наименование объекта',f'Корпус {no}')
        field('9.2.3','Регион','Москва')
        field('9.2.20','Максимальное количество этажей','12')
        field('9.2.21','Общая площадь здания', '10 000,00 м2' if no==1 else '20 000,00 м2')
    c.showPage();page()
    for no in range(1,3 if second else 2):
        field('9.3.1','Площадь жилых помещений','6 000,00 м2' if no==1 else '12 000,00 м2')
        field('9.3.2','Площадь нежилых помещений','1 000,00 м2' if no==1 else '2 000,00 м2')
        field('9.3.3','Общая площадь помещений','7 000,00 м2' if no==1 else '14 000,00 м2')
    c.showPage();page()
    for no in range(1,3 if second else 2):
        text(f'Объект № {no}')
        field('10.6.1','Наименование жилого комплекса','Учебный ЖК «Сад»')
        field('18.1.1','Планируемая стоимость строительства','1 000,00 млн руб.' if no==1 else '3 000 000,00 тыс. руб.')
        y-=25
    c.save();data=buffer.getvalue()
    if scan:
        original=fitz.open(stream=data,filetype='pdf');out=fitz.open()
        for p in original:
            image=p.get_pixmap(matrix=fitz.Matrix(2.5,2.5),alpha=False)
            new=out.new_page(width=p.rect.width,height=p.rect.height)
            new.insert_image(new.rect,stream=image.tobytes('png'))
        data=out.tobytes(deflate=True);out.close();original.close()
    return data
