import csv
import os
import time
from datetime import datetime
from time import sleep
import random
import math
from scipy.stats import logistic
import pandas as pd
import numpy as np
from flask import Flask, render_template, redirect, url_for, request, stream_with_context, Response
from flask_wtf import FlaskForm
from wtforms import SelectField, StringField, IntegerField, SubmitField, RadioField
from wtforms.validators import DataRequired
import trainModel as tm
# util functions
from bowler_list import bowler_list
from batsman_list import batsman_list
from team_list import team_list
from predict import Myrun
from datastreaming import Id_list
if random.random()>0.7:
    tm.main()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hello its me'

# create app instance
app = Flask(__name__)
app.config.from_object(Config)

# some init functions
team_list = team_list()
over_list = range(1,16)
totWick_list = range(0,9)
overLastWick_list = range(1,15)

bowler_list = bowler_list()
batsman_list = batsman_list()

#TODO:
#sort batsman, bowler, team list
#over of last wicket should be less than or equal to current over
#if tot wickets is 0, over of last wicketshould be disabled
#if bowler batsman combi is not correct acc to data, err
#if batsman should not be same as non-striker

# forms

class MakeTeamForm(FlaskForm):
    #TODO: both need to be radio buttons
    #TODO: disable newteam for now
    preteam =RadioField('Teams', choices=[(1,'Pre created teams')], default=1, coerce=int)
    #preteam =RadioField('Teams', choices=[(1,'Pre created teams'),(0,'Create new teams')], default=1, coerce=int)
    submit = SubmitField('Submit')

class SelectTeamForm(FlaskForm):
    team1 =SelectField('Team 1', validators=[DataRequired()])
  #  team2 = SelectField('Team2',choices=[], validators=[DataRequired()])
    submit = SubmitField('Submit')

class SelectTeam2Form(FlaskForm):
#    team1 =SelectField('Team 1', validators=[DataRequired()])
    team2 = SelectField('Team2', validators=[DataRequired()])
    pitch = SelectField('Pitch (5: Baller favouring)', validators=[DataRequired()])
    weather = SelectField('Weather (5: Baller favouring)', validators=[DataRequired()])
    submit = SubmitField('Submit')

    
# routes
@app.route('/', methods=['GET', 'POST'])    
def home():
    form = MakeTeamForm(request.form)
    #TODO: preteam=1 by default for now
    #TODO: newteam =0 by default for now
    if form.is_submitted():
        #print("preteam value")
        #preteam=form.preteam.data
        print(form.errors)
        return redirect(url_for('team', preteam=form.preteam.data))
    return render_template('home.html', form=form)

def find_team_list(t):
    #read pair csv file
    data = pd.read_csv('ListAllPairs.csv')
    #for val of t in col1 extract list of elements from col2
    t2List=data[(data["batting_team"]== t)]['bowling_team'].unique().tolist() 
    t2List.sort()
    return t2List
	
@app.route('/team/<preteam>', methods=['GET', 'POST'])    
def team(preteam):
    #TODO: change according to preteam value. Is 1 now
    #print("preteam value")
    #print(preteam)
    form = SelectTeamForm(request.form)
    form.team1.choices = [(team) for team in team_list]
   # form.team2.choices = [(team) for team in find_team_list(team_list[0])]
    if form.is_submitted():
        print(form.errors)
        return redirect(url_for('team2', team1=form.team1.data))
    return render_template('team.html', form=form, preteam=preteam)
def caly(pitch,weather):

    y = random.random()
    y=y*((int(pitch)+int(weather))/2)   
    y=logistic.cdf(y)
    if y>=0.5:
        return 1
    else:
        return 0
    
def write_data(row):

    fields = ["inning","match_id","TotalWicketsTillNow","TotalRunsTillNow","OverLastWicket","LastBallNo","LastOverNo"]
    
    # name of csv file  
    filename = "ComputedData.csv"

    # writing to csv file  
    with open(filename, 'w') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  

        # writing the fields  
        csvwriter.writerow(fields)  
        csvwriter.writerow(row)
        
@app.route('/team2/<team1>', methods=['GET', 'POST'])    
def team2(team1):
    #TODO: change according to preteam value. Is 1 now
    #print("preteam value")
    #print(preteam)
    form = SelectTeam2Form(request.form)
    form.pitch.choices = [1,2,3,4,5]
    form.weather.choices = [1,2,3,4,5]
    form.team2.choices = [(team) for team in find_team_list(team1)]
    if form.is_submitted():
        print(form.errors)
        return redirect(url_for('predict', team1=team1, team2=form.team2.data, pitch=form.pitch.data, weather=form.weather.data))
    return render_template('team2.html', form=form, team1=team1)

def stream_template(template_name, **context):
    # http://flask.pocoo.org/docs/patterns/streaming/#streaming-from-templates
    app.update_template_context(context)
    t = app.jinja_env.get_template(template_name)
    rv = t.stream(context)
    # uncomment if you don't need immediate reaction
    ##rv.enable_buffering(5)
    return rv


@app.route('/predict/<team1>_<team2>_<pitch>_<weather>', methods=['GET', 'POST'])
def predict(team1, team2,pitch,weather):
    def g():
        [inning, match_id]=Id_list(team1,team2)
        data=pd.read_csv("ballByballData.csv",low_memory=False)
        stream=data[(data["match_id"]== match_id)][['batting_team','bowling_team','inning', 'ball', 'over', 'batsman','non_striker','bowler','total_runs','player_dismissed']]
        stream.index = np.arange(0, len(stream) )
        old_inning=1
        tot_wick=0
        sum_runs=0
        over_Lastwick=0
        last_over=0
        p1=0.0
        p2=0.0
        p3=0.0
        p4=0.0
        for index, row in stream.iterrows():
            if (old_inning != row['inning']):
                old_inning=row['inning']
                tot_wick=0
                sum_runs=0
                over_Lastwick=0
                last_over=0
                p1=0.0
                p2=0.0
                p3=0.0
                p4=0.0
            #print("inning", inning, "tot_wick", tot_wick, "sum runs", sum_runs, "last_over", last_over, "over_Lastwick", over_Lastwick)
            #yield index, row['over']
            sum_runs += row['total_runs']
            if (last_over != row['over']):
                #print("calling Myrun", "bowler", row['bowler'], "over", last_over)
                last_over=row['over']
                
            if (last_over<17):
                p1=Myrun(row['bowler'], row['batsman'], row['non_striker'], int(last_over),tot_wick,over_Lastwick,1,[caly(pitch,weather)])
                p2=Myrun(row['bowler'], row['batsman'], row['non_striker'], int(last_over),tot_wick,over_Lastwick,2,[caly(pitch,weather)])
                p3=Myrun(row['bowler'], row['batsman'], row['non_striker'], int(last_over),tot_wick,over_Lastwick,3,[caly(pitch,weather)])
                p4=Myrun(row['bowler'], row['batsman'], row['non_striker'], int(last_over),tot_wick,over_Lastwick,4,[caly(pitch,weather )])

            pldis=row['player_dismissed']
            if pldis is np.nan :
                pldis="None"
            else:
                tot_wick += 1
                over_Lastwick=row['over']

            yield int(row['inning']),row['batting_team'],row['bowling_team'],int(row['over']),int(row['ball']), row['batsman'], row['non_striker'], row['bowler'], int(row['total_runs']),pldis,int(sum_runs),int(tot_wick),int(over_Lastwick),p1,p2,p3,p4

            sleep(.1)
    return Response(stream_template('predict.html', data=g()))


'''
@app.route('/predict/<team1>_<team2>', methods=['GET', 'POST'])
def predict(team1, team2):
    def generate():
        tot_wick=0
        sum_runs=0
        over_Lastwick=0
        [inning, match_id]=Id_list(team1,team2)
        data=pd.read_csv("ballByballData.csv",low_memory=False)
        stream=data[(data["match_id"]== match_id)][['batting_team','bowling_team','inning', 'ball', 'over', 'batsman','non_striker','bowler','total_runs','player_dismissed']]
        stream.index = np.arange(0, len(stream) )
        inning=1
        last_over=0
        p1=0.0
        p2=0.0
        p3=0.0
        p4=0.0
        for index, row in stream.iterrows():
            sum_runs += row['total_runs']
            if (last_over != row['over']):
                last_over=row['over']
                p1=Myrun(row['batsman'], row['bowler'], row['non_striker'], int(last_over),tot_wick,over_Lastwick,1)
                p2=Myrun(row['batsman'], row['bowler'], row['non_striker'], int(last_over),tot_wick,over_Lastwick,2)
                p3=Myrun(row['batsman'], row['bowler'], row['non_striker'], int(last_over),tot_wick,over_Lastwick,3)
                p4=Myrun(row['batsman'], row['bowler'], row['non_striker'], int(last_over),tot_wick,over_Lastwick,4)
    
            if row['player_dismissed'] is np.nan :
                yield "<p><p><b>Inning</b>: {} <b>Batting_team</b>: {} <b>Bowling_team</b>: {} <b>Over</b>: {} <b>Ball</b>: {} <b>Batsman</b>: {} <b>Non Striker</b>: {} <b>Bowler</b>: {} <b>Runs</b>: {}</p><p>**********<b>Total Runs</b>: {} <b>Total Wickets</b>: {} <b>Over of Last Wicket</b>: {}</p><p>**********<b>Probability of wicket in </b> <b>+1 Over</b>: {} <b>+2 Over</b>: {}<b>+3 Over</b>: {} <b>+4 Over</b>: {}</p><p>>>>>>>>>>>>>>>>>>>>>>></p></p>".format(int(row['inning']),row['batting_team'],row['bowling_team'],int(row['over']),int(row['ball']), row['batsman'], row['non_striker'], row['bowler'], int(row['total_runs']),int(sum_runs),int(tot_wick),int(over_Lastwick),p1,p2,p3,p4)
            else:
                tot_wick += 1
                over_Lastwick=row['over']
                yield "<p><p><b>Inning</b>: {} <b>Batting_team</b>: {} <b>Bowling_team</b>: {} <b>Over</b>: {} <b>Ball</b>: {} <b>Batsman</b>: {} <b>Non Striker</b>: {} <b>Bowler</b>: {} <b>Player dismissed</b>: {} </p> <p>**********<b>Total Runs</b>: {} <b>Total Wickets</b>: {} <b>Over of Last Wicket</b>: {}</p> <p>**********<b>Probability of wicket in </b> <b>+1 Over</b>: {} <b>+2 Over</b>: {}<b>+3 Over</b>: {} <b>+4 Over</b>: {}</p><p>>>>>>>>>>>>>>>>>>>>>>></p></p>".format(int(row['inning']),row['batting_team'],row['bowling_team'],int(row['over']),int(row['ball']), row['batsman'], row['non_striker'], row['bowler'], row['player_dismissed'],int(sum_runs),int(tot_wick),int(over_Lastwick),p1,p2,p3,p4)
            
    return Response(stream_with_context(generate()))
'''

'''
    #from pandas import read_csv
    
    sum_runs=0
    form = PredictionForm(request.form)
    if form.is_submitted():

        FxData=pd.read_csv("ComputedData.csv",low_memory=False) 
        inning=FxData.loc[0,'inning']
        match_id=FxData.loc[0,'match_id']
        LastBallNo=FxData.loc[0,'LastBallNo']
        over=FxData.loc[0,'OverLastWicket']
        data=pd.read_csv("ballByballData.csv",low_memory=False)
        #make batsman, bowler list based on team1, team2 ???
        #these lists will change when innings change ???
      #  match_id,inning,batting_team,bowling_team,over,ball,batsman,non_striker,bowler,is_super_over,wide_runs,bye_runs,legbye_runs,noball_runs,penalty_runs,batsman_runs,extra_runs,total_runs,player_dismissed,dismissal_kind,fielder,,over,Tot_runs,run_1,,out_over,out_over_1
        print("inning", inning)
        print("match_id",match_id)
            if (LastBallNo>5):
                over=over+1
                ball=1
            else:
                ball=LastBallNo+1
            #fields = ["inning","match_id","TotalWicketsTillNow","TotalRunsTillNow","OverLastWicket","LastBallNo","LastOverNo"]
            if (over>20):
                inning=inning%2 + 1
            print("ball", ball)
            print("over", over)
            print("inning", inning)
            stream=data[(data["match_id"]== match_id)&(data["inning"]== inning)&(data["over"]== over)&(data["ball"]== ball)][['batsman','non_striker','bowler','total_runs','player_dismissed']]
            stream.index = np.arange(0, len(stream) )
        #sum_runs+=stream.loc[0,'total_runs']
        #[['batsman','non_striker','bowler','over']]
        pd.set_option("display.max_rows", 200, "display.max_columns", 5)
        print("player_dismissed",stream.loc[0,'player_dismissed'])
    #   'plus1': Myrun(form.batsman.data, form.bowler.data, form.batsman_ns.data, form.over.data, form.totWick.data, form.overLastWick.data,1),
     #   print(stream.iloc[0,1])  
      #  print(stream.loc[0,'batsman'])
        
        #form.batsman.choices = [(batsman) for batsman in batsman_list]
        #form.batsman_ns.choices = [(batsman) for batsman in batsman_list]
        #form.bowler.choices = [(bowler) for bowler in bowler_list]
        #form.over.choices = [(over) for over in over_list]
        #form.totWick.choices = [(totWick) for totWick in totWick_list]
        #form.overLastWick.choices = [(overLastWick) for overLastWick in overLastWick_list]
        predictions = {
        'batTeam':{'ans':team1},
        'bowlTeam':{'ans':team2},
        'batsman':{'ans':stream.loc[0,'batsman']},
        'bowler':{'ans':stream.loc[0,'bowler']},
        'nstriker':{'ans':stream.loc[0,'non_striker']},
        'overno':{'ans':over},
        'ballno':{'ans':ball},
        'run':{'ans':stream.loc[0,'total_runs']},
        'out':{stream.loc[0,'player_dismissed']},
        'runs':{'ans':sum_runs},
        'wickets':{'ans':0},
        'lwicket':{'ans':0},
        'plus1':{'ans':0},
        'plus2':{'ans':0},
        'plus3':{'ans':0},
        'plus4':{'ans':0},
        
        #'plus1': Myrun(batsman1, bowler1, non_striker, over1,totWick1,overLastWick1,1),
        #'plus2': Myrun(batsman1, bowler1, non_striker, over1,totWick1,overLastWick1,2),
        #'plus3': Myrun(batsman1, bowler1, non_striker, over1,totWick1,overLastWick1,3),
        #'plus4': Myrun(batsman1, bowler1, non_striker, over1,totWick1,overLastWick1,4),
        }  
        rendered1=render_template('predict.html', form=form, predictions=predictions)
        time.sleep(5)
        return redirect(url_for('predict', team1=team1, team2=team2)) 
    return render_template('predict.html', form=form, team1=team1, team2=team2)
'''


# start server
if __name__ == '__main__':
    #print(batsman_list)
    #print(bowler_list)
    app.run(host= '127.0.0.1', port=5050, debug=False)

