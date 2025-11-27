import os
import cv2
import cxas
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from io import BytesIO
from PIL import Image, ImageFilter

class Anatomy:
	color = (255, 99, 132) # Soft Red
	def __init__(self, image, label, mask=None, contour=None, color=None):
		self.image = image
		if color is not None:
			self.color = color
		
		if image is None and contour is None:
			raise AttributeError("At least one of mask or contour is required to create an anatomy")
		
		if mask is None:
			mask = np.zeros(self.image.size[::-1], dtype=np.uint8)
			cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
		self.mask = mask
		self.area = int((self.mask.astype(np.float32) / 255).sum())
		
		if contour is None:
			if self.mask.astype(bool).any():
				contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				contour = max(contours, key=cv2.contourArea)
			else:
				contour = np.array([[-1, -1]])
		self.contour = contour
		x, y, self.width, self.height = cv2.boundingRect(self.contour)
		
		self.label = label
		
	@property
	def nickname(self):
		if len(self.label) <= 7 or len(self.label.split()) == 1:
			return self.label
		words = self.label.split()
		return "".join(x[0].upper() for x in words)
	
	@property
	def bbox(self):
		return Image.fromarray(self.mask).convert("L").getbbox()
	
	@property
	def center(self):
		M = cv2.moments(self. contour)
		
		if M["m00"] != 0:
			cx = M["m10"] / M["m00"]
			cy = M["m01"] / M["m00"]
		else:
			# Contour area is zero (degenerate case)
			cx, cy = 0, 0
			
		return int(cx), int(cy)

	@property
	def contour_image(self):
		contour_image = np.zeros((*self.image.size[::-1], 4), dtype=np.uint8)
		if self.mask.max() == 0: # Empty mask
			return contour_image
		cv2.drawContours(contour_image, [self.contour], -1, (*self.color, 255), thickness=2)
		contour_image = Image.fromarray(contour_image)
		return contour_image

	@property
	def label_image(self):
		return self.get_label_image(label=self.nickname)
	
	def get_label_image(self, label=None):
		if label is None:
			label = self.nickname
		label_image = np.zeros((*self.image.size[::-1], 4), dtype=np.uint8)
		if self.mask.max() == 0: # Empty mask
			return label_image
		
		font = cv2.FONT_HERSHEY_SIMPLEX
		font_scale = (0.4) * (self.image.size[0] / 512) + 0.1
		font_thickness = 1
		
		# points = self.contour.squeeze()
		# x, y = points[np.random.choice(len(points))]
		x, y = self.contour.squeeze()[0]
		(text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
		
		margin = 2
		width, height = self.image.size
		x_min, y_min, x_max, y_max =  x - margin, y - text_height - margin, x + text_width + margin, y + margin
		if x_min < 0:
			x = margin + int(text_width / 2 + 2)
		elif x_max > width:
			x = width - margin - int(text_width / 2 + 2)
		if y_min < 0:
			y = margin + int(text_height / 2 + 2)
		elif y_max > height:
			y = height - margin - int(text_height / 2 + 2)
		x_min, y_min, x_max, y_max =  x - margin, y - text_height - margin, x + text_width + margin, y + margin
		
		cv2.rectangle(label_image, (x_min, y_min), (x_max, y_max), (*self.color, 255), thickness=-1) # -1 -> filled
		cv2.putText(label_image, label, (x, y), font, font_scale, (255, 255, 255, 255), font_thickness, lineType=cv2.LINE_AA)
		
		label_image = Image.fromarray(label_image)
		return label_image
	
	def get_random_point(self, n=1, kernel_size=None, margin=None):
		mask = self.mask
		if kernel_size is not None:
			if margin is None:
				margin = (kernel_size[0] // 5, kernel_size[1] // 5)
			kernel_size =(kernel_size[0] - margin[0], kernel_size[1] - margin[1])
			kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
			mask = cv2.erode(self.mask, kernel)
			
		if not mask.any():
			return None
		
		ys, xs = np.where(mask)
		points = np.stack((xs, ys), axis=1)
		random_point = points[np.random.choice(len(points), size=n)]

		if n == 1:
			return random_point[0]
		return random_point
	
	def convex(self):
		if self.mask.astype(bool).any():
			contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			contour = max(contours, key=cv2.contourArea)
			contour = cv2.convexHull(contour)
		else:
			contour = np.array([[-1, -1]])        
		
		return Anatomy(self.image, self.label, contour=contour, color=self.color)
	
	def _repr_png_(self):
		"""iPython display hook support

		:returns: png version of the image as bytes
		"""
		_repr_img = self.image.convert("RGB")
		if self.mask.max() > 0:
			_repr_img.paste(self.contour_image, (0, 0), mask=self.contour_image)
			_repr_img.paste(self.label_image, (0, 0), mask=self.label_image)
		
		b = BytesIO()
		try:
			_repr_img.save(b, format="PNG")
		except Exception as e:
			msg = "Could not save to PNG for display"
			raise ValueError(msg) from e
		return b.getvalue()
	
	def __sub__(self, other):
		if not isinstance(other, Anatomy) and self.image == other.image:
			return NotImplemented("Only support subtraction between two anatomy of same CXR")
		
		mask = cv2.subtract(self.mask, other.mask)
		
		return Anatomy(self.image, self.label, mask=mask, color=self.color)
	
	def __isub__(self, other):
		self = self - other
		return self
	
	def __add__(self, other):
		if not isinstance(other, Anatomy) and self.image == other.image:
			return NotImplemented("Only support addition between two anatomy of same CXR")
		
		mask = cv2.add(self.mask, other.mask)
		
		return Anatomy(self.image, self.label, mask=mask, color=self.color)
	
	def __isadd__(self, other):
		self = self + other
		return self
	
	def __contains__(self, item):
		"""
		item (x, y) -> check if point in mask
		item (np.array/PIL.Image) -> check if array image fit in mask (over 95% coverage)
		"""
		if isinstance(item, tuple):
			result = cv2.pointPolygonTest(self.contour, item, measureDist=False)
			return (int(result) + 1) / 2
		
		if isinstance(item, Image.Image):
			if item.size != self.image.size:
				raise NotImplemented("Only support PIL.Image.Image of same size as Anatomy.image")
			item = np.array(item.convert("L"), dtype=np.float32) / 255
		elif isinstance(item, np.ndarray):
			if item.min() < 0 and item.max() > 0:
				raise NotImplemented("Only support np.ndarray of grayscale between [0, 1] and of same size as Anatomy.image")
			item = item.astype(np.float32)
		else:
			raise NotImplemented("Only support tuple, PIL.Image.Image or np.ndarray")
		
		covered = ((self.mask).astype(np.float32) / 255) * item
		return covered.sum() / item.sum() > 0.95

	def resize(self, size):
		image = self.image.resize(size)
		mask = np.array(Image.fromarray(self.mask).resize(size))
		return Anatomy(image, self.label, mask=mask, color=self.color)


colors = [
	(255, 99, 132),    # Soft Red
	(54, 162, 235),    # Sky Blue
	(255, 206, 86),    # Soft Yellow
	(75, 192, 192),    # Aqua
	(153, 102, 255),   # Lavender
	(255, 159, 64),    # Orange
	(255, 105, 180),   # Hot Pink
	(0, 191, 255),     # Deep Sky Blue
	(144, 238, 144),   # Light Green
	(255, 182, 193),   # Light Pink
	(0, 255, 127),     # Spring Green
	(186, 85, 211),    # Medium Orchid
	(100, 149, 237),   # Cornflower Blue
	(240, 230, 140),   # Khaki
	(255, 228, 225),   # Misty Rose
	(135, 206, 250),   # Light Sky Blue
	(221, 160, 221),   # Plum
	(175, 238, 238),   # Pale Turquoise
	(152, 251, 152),   # Pale Green
	(255, 222, 173),   # Navajo White
	(255, 218, 185),   # Peach Puff
	(216, 191, 216),   # Thistle
	(173, 216, 230),   # Light Blue
	(255, 240, 245),   # Lavender Blush
	(250, 250, 210),   # Light Goldenrod Yellow
	(176, 224, 230),   # Powder Blue
	(255, 250, 205),   # Lemon Chiffon
	(238, 130, 238),   # Violet
	(255, 215, 0),     # Gold
	(60, 179, 113),    # Medium Sea Green
]

import numpy as np
from PIL import Image
class CXRImage:
	def __init__(self, image, lung_label=None):
		self.i = 0 # for color

		if isinstance(image, np.ndarray):
			self.image = Image.fromarray(image)
		elif isinstance(image, Image.Image):
			self.image = image
		else:
			raise NotImplementedError("Image should be of type np.ndarray or PIL.Image.Image")

		self.labels = []
		
		if lung_label is None:
			right_lung_mask, left_lung_mask = self.segment_lungs()
		else:
			right_lung_mask, left_lung_mask = lung_label == 1, lung_label == 2

		self.add_mask_as_anatomy("right lung", right_lung_mask)
		self.add_mask_as_anatomy("left lung", left_lung_mask)

	@property
	def np_image(self):
		return np.array(self.image)

	def add_mask_as_anatomy(self, label, bin_mask):
		attr_name = "_".join(label.split())
		mask = bin_mask.astype(np.uint8) * 255
		color = colors[self.i%len(colors)]
		setattr(self, attr_name, Anatomy(self.image, label, mask=mask, color=color))
		
		self.labels.append(label)
		self.i += 1

	def segment_lungs(self):
		from lung_utils import brighten_border, lung_segmentation
		image = brighten_border(self.image)
		right_lung_mask, left_lung_mask = lung_segmentation(image)

		return right_lung_mask, left_lung_mask

	@property
	def size(self):
		return self.image.size
	
	@property
	def width(self):
		return self.image.width
	
	@property
	def height(self):
		return self.image.height
	
	@property
	def anatomy_attrs(self):
		return ["_".join(label.split()) for label in self.labels]
	
	@property
	def anatomies(self):
		return [getattr(self, anatomy_attr) for anatomy_attr in self.anatomy_attrs]
	
	def resize(self, size):
		self.image = self.image.resize(size)
		for attr_name, anatomy in zip(self.anatomy_attrs, self.anatomies):
			setattr(self, attr_name, anatomy.resize(size))
		return self
	
	def percentile(self, percentage):
		flattened_image = np.array(self.image).flatten()
		return int(np.percentile(flattened_image, percentage))

	def pil_image(self, contour=True, label=True):
		_repr_img = self.image.convert("RGB")
		if contour:
			for anatomy in self.anatomies:
				if anatomy.mask.max() > 0:
					_repr_img.paste(anatomy.contour_image, (0, 0), mask=anatomy.contour_image)
		if label:
			for anatomy in self.anatomies:
				if anatomy.mask.max() > 0:
					_repr_img.paste(anatomy.label_image, (0, 0), mask=anatomy.label_image)
		return _repr_img
			
	def _repr_png_(self):
		"""iPython display hook support

		:returns: png version of the image as bytes
		"""
		
		b = BytesIO()
		try:
			self.pil_image().save(b, format="PNG")
		except Exception as e:
			msg = "Could not save to PNG for display"
			raise ValueError(msg) from e
		return b.getvalue()

import cv2
import random
import numpy as np

from PIL import Image, ImageFilter, ImageChops

class Brush:
	def __init__(self, filepath=None, image=None, color=(135, 206, 250)):
		self.color = color
		
		if filepath is None and image is None:
			raise AttributeError("At least One of filepath or image (RGBA) needs to be given")
		elif filepath is not None and image is not None:
			raise AttributeError("Only One of filepath or image (RGBA) can be given")
		elif filepath is None and (not isinstance(image, Image.Image) or image.mode != "RGBA"):
			raise AttributeError("Image given needs to be PIL.Image.Image in RGBA mode")
		
		if filepath is not None:
			raw_image = Image.open(filepath).convert("L")
			self.image = Image.new("RGBA", raw_image.size, (*color, 255))
			self.image.putalpha(raw_image)
		else:
			self.image = Image.new("RGBA", image.size, (*color, 255))
			self.image.putalpha(image.getchannel("A"))
			
		bbox = self.mask.getbbox()
		self.image = self.image.crop(bbox)
			
	@property
	def mask(self):
		return self.image.getchannel("A")
	
	@property
	def alpha(self):
		return self.image.getchannel("A")
	
	@property
	def width(self):
		return self.image.width
		
	@property
	def height(self):
		return self.image.height
	
	@property
	def size(self):
		return self.width, self.height
	
	@property
	def ratio(self):
		return self.width / self.height
			
	def _repr_png_(self):
		"""iPython display hook support

		:returns: png version of the image as bytes
		"""        
		b = BytesIO()
		try:
			self.image.save(b, format="PNG")
		except Exception as e:
			msg = "Could not save to PNG for display"
			raise AttributeError(msg) from e
		return b.getvalue()
	
	def resize(self, width=None, height=None):
		if width is None and height is None:
			raise AttributeError("At least one of width or height needs to be given")
		
		# Both width and height are given -> set directly            
		# Only height or width is given -> in ratio resize
		if width is None and height is not None:
			width = max(int(height * self.ratio), 1)
		elif height is None and width is not None:
			height = max(int(width / self.ratio), 1)
		
		image = self.image.resize((width, height))
		return Brush(image=image, color=self.color)
	
	def rotate(self, angle):
		image = self.image.rotate(angle, resample=Image.BICUBIC, expand=True)
		return Brush(image=image, color=self.color)
	
	def change_opacity(self, opacity):
		image = self.image.copy()
		image.putalpha(self.alpha.point(lambda p: int(p * opacity)))
		return Brush(image=image, color=self.color)
	
	def copy(self):
		# Deep copy mask and image
		return Brush(image=self.image.copy(), color=self.color)
	
	@staticmethod
	def brush_jitter(brush):
		base_width = brush.width
		brush_setting = {
			"width_range": (int(base_width * 0.3), base_width), # size in pixels
			"angle_range": (0, 360),
			"opacity_range": (0.07, 0.3),
			"flow_range": (0.67, 1.0),
		}
		
		# Get random brush properties
		width = np.random.randint(*brush_setting["width_range"])
		angle = np.random.uniform(*brush_setting["angle_range"])
		opacity = np.random.uniform(*brush_setting["opacity_range"])
		flow = np.random.uniform(*brush_setting["flow_range"])

		# Setup according to properties (used to define brush alpha)
		brush = brush.resize(width=width)
		brush = brush.rotate(angle)
		brush = brush.change_opacity(opacity * flow)

		return brush

