from flask import Flask, render_template, request, redirect, session
 
app = Flask(__name__)

@app.route("/")
def hello():
    return "Le serveur flask fonctionne"
    
    
if __name__ == '__main__':
    app.run(port=7318)