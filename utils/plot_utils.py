import numpy as np
import sys
from typing import List, Dict, Tuple

from PIL import ImageDraw, Image, ImageFont


def scale_colors(images: np.ndarray, nchannels=3):
  try:
    orig_shape = images.shape
    assert images.shape[-1] == nchannels

    if images.dtype == np.uint:
      return images

    from scipy.misc import toimage, fromimage
    def prep(img):
      return fromimage(toimage(np.clip(img, 0, 255), mode='RGB'))

    if images.ndim == 3:
      return fromimage(toimage(np.clip(images, 0, 255)))

    images = images.copy()
    res = np.zeros(images.shape, dtype=np.uint8)
    res.shape = (-1,) + images.shape[-nchannels:]
    images.shape = res.shape
    N = res.shape[0]

    for i in range(N):
      res[i] = prep(images[i])

    res.shape = orig_shape
    return res
  except:
    print('input shape: %s' % orig_shape, file=sys.stderr)
    raise


def len_none(o) -> int:
  if o is None:
    return 0
  return len(o)

def get_image_shapes(images: List[List[str]],
                    N=-1, M=-1)-> np.ndarray:
  """
  Computes a full array of text sizes as H, W vectors
  :param images: np.ndarray or list of lists holding the images. If this is a list of lists, the result will hold zeros for the missing elements to make a full array.
  :param M: minimum column count for output array. Will be filled with zeros if necessary.
  :param N: minimum row count for output array. Will be filled with zeros if necessary.
  :return: an np.ndarray of shape [N, M, 3], holding H, W, C for each given image.

  """

  N = max(N, len_none(images))
  M = max(M, max(map(len_none, [] if images is None else images), default=0))

  # if we have a proper ndarray, we know all the sizes immediately
  if isinstance(images, np.ndarray) and images.dtype != np.object_:
    shape = images.shape[2:]
    return np.tile([[shape]], [N, M, 1])

  result = np.zeros((N, M, 3), dtype=np.int64)

  if images is None:
    return result

  for ir, row in enumerate(images):
    if row is None: continue
    for ic, im in enumerate(row):
      if im is not None:
        if im.ndim not in (2, 3):
          raise ValueError('Image (%i, %i) has invalid number of dimensions %i. Must be 2 for grayvalue or 3 for RGB[A].' % (ir, ic, im.ndim))
        if im.ndim == 3 and im.shape[2] not in [1, 3, 4]:
          raise ValueError('Image (%i, %i) as invalid number of channels %i. Must be 1, 3, or 4.' % (ir, ic, im.shape[2]))
        result[ir, ic, :2] = im.shape[:2]
        result[ir, ic, 2] = im.shape[2] if im.ndim == 3 else 1

  return result


def get_text_sizes(titles: List[List[str]], font,
                   N=-1, M=-1)-> np.ndarray:
  """
  Computes a full array of text sizes as H, W vectors
  :param titles: Array or list of lists holding the titles. If this is a list of lists, the result will hold zeros for the missing elements to make a full array.
  :param M: minimum column count for output array. Will be filled with zeros if necessary.
  :param N: minimum row count for output array. Will be filled with zeros if necessary.
  :return: an np.ndarray of shape [N, M, 2], holding H, W for each given title

  """

  N = max(N, len_none(titles))
  M = max(M, max(map(len_none, [] if titles is None else titles), default=0))

  result = np.zeros((N, M, 2), dtype=np.int64)
  textsize_im = ImageDraw.Draw(Image.new(mode='RGB', size=(1, 1)))

  if titles is None:
    return result

  for ir, row in enumerate(titles):
    if row is None: continue
    for ic, t in enumerate(row):
      if t is not None:
        result[ir, ic] = textsize_im.textsize(titles[ir][ic], font=font)[::-1]

  return result


def image_grid(images: List[List[np.ndarray]],
               titles: List[List[str]]=None,
               margin_h=5,
               margin_h_text=5,
               margin_w=5,
               uniform_row_height=False,
               uniform_col_width=False,
               font_size=22,
               margin_after_first_col=0) -> Image:
  font = ImageFont.truetype("DejaVuSans.ttf", font_size)

  n_row = max(len_none(images),
              len_none(titles))
  n_col = max(max(map(len_none, [] if images is None else images), default=0),
              max(map(len_none, [] if titles is None else titles), default=0))

  # the (minimum) required text height and width for every item
  text_cell_hw = get_text_sizes(titles, font, N=n_row, M=n_col)

  # shape of each image in order H, W, C
  image_shape = get_image_shapes(images, N=n_row, M=n_col)

  if uniform_row_height:
    text_row_height = np.amax(text_cell_hw[:, :, 0]) * np.ones(n_row, dtype=np.int64)
    im_row_height = np.amax(image_shape[:, :, 0]) * np.ones(n_row, dtype=np.int64)
  else:
    text_row_height = np.amax(text_cell_hw[:, :, 0], axis=1)
    im_row_height = np.amax(image_shape[:, :, 0], axis=1)

  if uniform_col_width:
    col_width = max(np.amax(text_cell_hw[:, :, 1]),
                    np.amax(image_shape[:, :, 1])) * np.ones(n_col, dtype=np.int64)
  else:
    col_width = np.maximum(np.amax(text_cell_hw[:, :, 1], axis=0),
                           np.amax(image_shape[:, :, 1], axis=0))

  # only use a vertical text margin for rows that actually have text
  margin_h_text = margin_h_text * (text_row_height != 0)

  H_total = im_row_height.sum() + text_row_height.sum() + (n_row+1) * margin_h + margin_h_text.sum()
  W_total = col_width.sum() + (n_col + 1) * margin_w + margin_after_first_col

  channels = np.amax(image_shape[:, :, 2])
  if channels == 4:
    res = Image.new(mode='RGBA', size=(W_total, H_total), color=(255, 255, 255))
  elif channels == 3:
    res = Image.new(mode='RGB', size=(W_total, H_total), color=(255, 255, 255))
  else:
    assert channels == 1, 'invalid number of channels %i. Must be 1, 3, or 4' % channels
    channels = 0
    res = Image.new(mode='L', size=(W_total, H_total), color=(255, 255, 255))
  d = ImageDraw.Draw(res)

  px_row = 0
  for ir in range(n_row):
    px_row += margin_h

    # draw images for this row
    px_col = 0
    for ic in range(n_col):
      px_col += margin_w
      if ic == 1:
        px_col += margin_after_first_col

      if image_shape[ir, ic].any():
        H, W = image_shape[ir, ic, :2]
        # find coords to center image in its cell
        im_pad_h = (im_row_height[ir] - H) // 2
        im_pad_w = (col_width[ic] - W) // 2
        try:
          im = get_pastable_image(images[ir][ic], channels)
        except:
          print('Problem with input at [%i, %i] of type %s and shape %s' % (ir, ic, images[ir][ic].dtype, images[ir][ic].shape))
          raise
        res.paste(im, (px_col + im_pad_w, px_row + im_pad_h))

      px_col += col_width[ic]

    px_row += im_row_height[ir] + margin_h_text[ir]

    # draw texts for this row
    px_col = 0
    for ic in range(n_col):
      px_col += margin_w

      if text_cell_hw[ir, ic].any():
        # find start coords for text
        tx_h, tx_w = text_cell_hw[ir, ic]
        tx_pad_h = (text_row_height[ir] - tx_h) // 2  # TODO: add an option to NOT center texts?
        tx_pad_w = (col_width[ic] - tx_w) // 2
        d.text((px_col + tx_pad_w, px_row + tx_pad_h), titles[ir][ic], fill=0, font=font)

      px_col += col_width[ic]

    px_row += text_row_height[ir]

  return res


def get_pastable_image(im: np.ndarray,
                       c_out: int):
  if c_out not in [1, 3, 4]:
    raise ValueError('Invalid number of channels %i. Must be 1, 3, or 4.' % c_out)
  if im.ndim not in [2, 3]:
    raise ValueError('Image has invalid number of dimensions %i. Must be 2 for grayvalue or 3 for RGB[A].' % im.ndim)
  if im.ndim == 3 and im.shape[2] > c_out:
    raise ValueError('Image with %i channels cannot automatically be converted to %i channels.' % (im.shape[2], c_out))

  if c_out == 1:
    if im.ndim == 3:
      return Image.fromarray(im[..., 0])
    else:
      return Image.fromarray(im)

  elif c_out == 3:
    if im.ndim == 2:
      im = im[..., np.newaxis]
    if im.shape[2] == 1:
      im = np.repeat(im, 3, axis=2)
    return Image.fromarray(im)

  else:  # c_out == 4
    if im.ndim == 3 and im.shape[2] == 4:
      return Image.fromarray(im)
    if im.ndim == 2:
      im = im[..., np.newaxis]
    if im.shape[2] == 1:
      im = np.repeat(im, 3, axis=2)
    im_rgba = np.ones(im.shape[:2]+(4,), dtype=np.uint8) * 255
    im_rgba[:, :, :3] = im
    return Image.fromarray(im_rgba)















