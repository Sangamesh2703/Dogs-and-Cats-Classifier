from fastai.vision.all import *
import gradio as gr

def is_cat(x): return x[0].isupper() 

#cell
im = PILImage.create('dog.jpg')
im.thumbnail((192,192))
im

#cell
learn = load_learner('model.pkl')
learn.predict(im)

#cell
categories = ('Dog','Cat')

def classify_images(img):
  pred,idx,probs = learn.predict(img)
  return dict(zip(categories, map(float,probs)))
  classify_images(im)

 #cell
image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()

intf = gr.Interface(fn=classify_images, inputs=image, outputs=label)
intf.launch(inline=False)

m = learn.model
ps = list(m.parameters())
ps[1]
ps[0].shape
ps[0]
