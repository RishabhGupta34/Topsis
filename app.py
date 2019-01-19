from flask import Flask,request,url_for,redirect,render_template
from pymongo import MongoClient
import cv2
import numpy as np
import pandas as pd
from skcriteria import Data, MIN, MAX
from skcriteria.madm import closeness, simple
from operator import itemgetter
import math

app=Flask(__name__)
app.secret_key = 'A?DSGREfgska[]dkoRERWF???::HLELFS'


MONGODB_URI = "mongodb://test:test12@ds159184.mlab.com:59184/topsis"
client = MongoClient(MONGODB_URI)
db = client.get_database("topsis")
user_data = db.user_data

def image_properties(image_name,image_data):
	nparr = np.fromstring(image_data, np.uint8)
	img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	clarity = cv2.Laplacian(gray, cv2.CV_64F).var()
	
	b=img[:,:,0]
	g=img[:,:,1]
	r=img[:,:,2]
	r=np.array(r,np.float32)
	g=np.array(g,np.float32)
	b=np.array(b,np.float32)
	gs = (np.sqrt(.241*(r**2)+.691*(g**2)+.068*(b**2)))
	Brightness = np.average(gs)
	
	dp=math.sqrt(img.shape[0]**2+img.shape[1]**2)
	Pixel=dp/5
	
	hist = cv2.calcHist([img], [0], None, [256], [0, 256])
	e=sum(hist*np.log2(hist))
	Contrast=e[0]
	    
	Resolution = min(img.shape[0],img.shape[1])
	
	rows, cols = img.shape[:2]
	kernel_x = cv2.getGaussianKernel(cols,200)
	kernel_y = cv2.getGaussianKernel(rows,200)
	kernel = kernel_y*kernel_x.T
	mask = 255 * kernel / np.linalg.norm(kernel)
	Vignette=np.amax(mask)
	data={
		'clarity':str(clarity),
		'Brightness':str(Brightness),
		'Pixel':str(Pixel),
		'Contrast':str(Contrast),
		'Resolution':str(Resolution),
		'Vignette':str(Vignette)
	}
	user_data.update_one({'Image_Name':image_name},{"$set":data})


@app.route('/leaderboard')
def leaderboard():
	s_name=[]
	s_roll=[]
	name=[]
	clarity=[]
	Brightness=[]
	Pixel=[]
	Contrast=[]
	Resolution=[]
	Vignette=[]
	for i in user_data.find():
		s_name.append(i['Name'])
		s_roll.append(i['Roll_No'])
		name.append(i['Image_Name'])
		clarity.append(float(i['clarity']))
		Brightness.append(float(i['Brightness']))
		Pixel.append(float(i['Pixel']))
		Contrast.append(float(i['Contrast']))
		Resolution.append(float(i['Resolution']))
		Vignette.append(float(i['Vignette']))
	df=pd.DataFrame({'image_name':name,'clarity': clarity,
           'Brightness': Brightness,'Pixel':Pixel,'Contrast':Contrast,'Resolution':Resolution,'Vignette':Vignette})
	criteria=[MAX,MAX,MAX,MAX,MIN,MAX]
	print(df)
	ds=np.array(df)
	ds1=ds[:,1:]
	data = Data(ds1, criteria,
            weights=[float(1.0)/6,float(1.0)/6,float(1.0)/6,float(1.0)/6,float(1.0)/6,float(1.0)/6],
            anames=ds[:,-1],
            cnames=["Brightness", "Contrast", "Pixel","Resolution","Vignette","Clarity"])
	t=closeness.TOPSIS()
	dec=t.decide(data)
	rank=dec.rank_
	y = rank.astype(np.float)
	topsis_score=dec.e_.closeness
	name=ds[:,0]
	result=[s_name,s_roll,y,topsis_score]
	result=np.array(result)
	result=result.T
	final=result[result[:,2].argsort()]
	return render_template("leaderboard.html",result=final)

@app.route('/home', methods=['GET', 'POST'])
def home():
	if request.method=='GET':
		return render_template('home.html')
	elif request.method=='POST':
		name=request.form.get('regname')
		email=request.form.get('regemail')
		rollno=request.form.get('rollno')
		image=request.files.get("image")
		img_data=image.read()
		if user_data.find_one({'Roll_No':rollno}):
			return render_template("home.html",error="Error! One user can submit only one entry")
		data={}
		data['Name']=name
		data['Image_Name']="_".join(name.split())+"_"+rollno
		data['Roll_No']=rollno
		data['Email']=email
		data['Image_data']=img_data
		user_data.insert_one(data)
		image_properties(data['Image_Name'],data['Image_data'])

		return render_template("home.html",error="Congrats! You have successfully submitted your entry. Check out the leaderboard for results")



if __name__=="__main__":
	app.run(port=8000,debug=True)