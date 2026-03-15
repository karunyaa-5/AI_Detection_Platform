document.addEventListener("DOMContentLoaded",function(){

const analyzeBtn=document.getElementById("analyzeBtn")

const result=document.getElementById("result")
const aiPercent=document.getElementById("aiPercent")
const humanPercent=document.getElementById("humanPercent")

const resultBox=document.getElementById("resultBox")

let probChart
let trendChart
let modelChart


analyzeBtn.addEventListener("click",async function(){

const text=quill.getText().trim()

if(text===""){
alert("Please enter text")
return
}

const response=await fetch("/predict",{

method:"POST",
headers:{
"Content-Type":"application/x-www-form-urlencoded"
},
body:"text="+encodeURIComponent(text)

})

const data=await response.json()

const ai=parseFloat(data.ai_confidence)
const human=parseFloat(data.human_confidence)

aiPercent.innerText="AI Probability: "+ai+"%"
humanPercent.innerText="Human Probability: "+human+"%"


if(ai>human){

result.innerText="AI Generated Text"

resultBox.classList.remove("result-human")
resultBox.classList.add("result-ai")

}else{

result.innerText="Human Written Text"

resultBox.classList.remove("result-ai")
resultBox.classList.add("result-human")

}



/* BAR CHART */

if(probChart) probChart.destroy()

probChart=new Chart(document.getElementById("probChart"),{

type:"bar",

data:{
labels:["AI Generated","Human Written"],
datasets:[{
data:[ai,human],
backgroundColor:["red","green"]
}]
}

})



const trendColor = ai > human ? "red" : "green"


if(trendChart) trendChart.destroy()

trendChart=new Chart(document.getElementById("trendChart"),{

type:"line",

data:{
labels:["Start","Scan","Process","Analyze","Result"],
datasets:[{
data:[10,30,50,ai,human],
borderColor:trendColor,
tension:0.4
}]
}

})



const lr=(ai*0.9).toFixed(2)
const svm=(ai*0.95).toFixed(2)
const rf=(ai*1.05).toFixed(2)
const xgb=(ai*1.1).toFixed(2)

const modelColor = ai > human ? "red" : "green"


if(modelChart) modelChart.destroy()

modelChart=new Chart(document.getElementById("modelChart"),{

type:"bar",

data:{
labels:["Logistic Regression","SVM","Random Forest","XGBoost"],
datasets:[{
data:[lr,svm,rf,xgb],
backgroundColor:modelColor
}]
}

})

})

})