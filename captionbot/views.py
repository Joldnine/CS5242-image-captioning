import requests
from django.shortcuts import render
from .services import CaptionMe
bot = CaptionMe()


def index(request, img_url=None):
    """
    params:
        img_url: could be 1) None, 2) 'random', 3) image url
    returns:
    """
    if not img_url:
        img_url = request.GET.get('url')
    if not img_url or not is_url_image(img_url):
        context = {
            'img_url': 'https://upload.wikimedia.org/wikipedia/commons/'\
                + 'thumb/a/ac/No_image_available.svg/1024px-No_image_available.svg.png',
            'caption': 'Image does not exits. Please make sure your url points to an image',
        }
        return render(request, 'captionbot/index.html', context)
    res, _ = bot.get_caption(img_url)
    res = ' '.join(res)
    context = {
        'img_url': img_url,
        'caption': res,
    }
    return render(request, 'captionbot/index.html', context)


def is_url_image(img_url):
    image_formats = ("image/png", "image/jpeg", "image/jpg")
    try:
        r = requests.head(img_url)
        if r.headers["content-type"] in image_formats:
            return True
    except requests.exceptions.MissingSchema:
        pass
    return False
