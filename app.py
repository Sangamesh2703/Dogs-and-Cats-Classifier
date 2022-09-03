from fastai.learner import load_learner
from PIL import Image
import gradio as gr

def is_cat(x): return x[0].isupper() 

#cell
im = PILImage.create('dog.jpg')
im.thumbnail((192,192))

#cell
learn = load_learner('model.pkl')
learn.predict(im)

#cell
categories = ('Dog','Cat')

def classify_images(img):
  pred,idx,probs = learn.predict(img)
  return dict(zip(categories, map(float,probs)))


#cell
image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()

intf = gr.Interface(fn=classify_images, inputs=image, outputs=label)
intf.launch(inline=False)

