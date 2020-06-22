import os
from flask import Flask, render_template, redirect, url_for, request
from flask_wtf import FlaskForm
from wtforms import SelectField, StringField, IntegerField, SubmitField, RadioField
from wtforms.validators import DataRequired

# util functions
from bowler_list import bowler_list
from batsman_list import batsman_list
from team_list import team_list
from predict import Myrun

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
    team2 = SelectField('Team2', validators=[DataRequired()])
    submit = SubmitField('Submit')

class PredictionForm(FlaskForm):
    
    batsman = SelectField('Batsman Name', validators=[DataRequired()])
    batsman_ns = SelectField('Non-Striker Batsman Name', validators=[DataRequired()])
    bowler = SelectField('Bowler Name', validators=[DataRequired()])
    over =SelectField('Current Over', coerce=int, validators=[DataRequired()])
    totWick =SelectField('Tot wicket till now', coerce=int, validators=[DataRequired()])
    overLastWick =SelectField('Over of last wicket', coerce=int, validators=[DataRequired()])
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

@app.route('/team/<preteam>', methods=['GET', 'POST'])    
def team(preteam):
    #TODO: change according to preteam value. Is 1 now
    print("preteam value")
    print(preteam)
    form = SelectTeamForm(request.form)
    form.team1.choices = [(team) for team in team_list]
    form.team2.choices = [(team) for team in team_list]

    if form.is_submitted():
        print(form.errors)
        return redirect(url_for('predict', team1=form.team1.data, team2=form.team2.data))
    return render_template('team.html', form=form, preteam=preteam)

@app.route('/predict/<team1>_<team2>', methods=['GET', 'POST'])
def predict(team1, team2):
    form = PredictionForm(request.form)
    #make batsman, bowler list based on team1, team2 ???
    #these lists will change when innings change ???

    form.batsman.choices = [(batsman) for batsman in batsman_list]
    form.batsman_ns.choices = [(batsman) for batsman in batsman_list]
    form.bowler.choices = [(bowler) for bowler in bowler_list]
    form.over.choices = [(over) for over in over_list]
    form.totWick.choices = [(totWick) for totWick in totWick_list]
    form.overLastWick.choices = [(overLastWick) for overLastWick in overLastWick_list]

    if form.is_submitted():
        print(form.errors)
        '''
        bowler1='R Bhatia'
        batsman1='TS Mills'
        non_striker='Yuvraj Singh'
        over1=5
        totWick1=1
        overLastWick1=1

        
        print(form.batsman.data)
        print(form.bowler.data)
        print(form.batsman_ns.data)
        print(form.over.data)
        print(form.totWick.data)
        print(form.overLastWick.data)

        
        bowler1='A Choudhary'
        batsman1='Yuvraj Singh'
        non_striker='MC Henriques'
        over1=4
        totWick1=2
        overLastWick1=3
        '''

        '''
        bowler1='A Nehra'
        batsman1='CH Gayle'
        non_striker='Mandeep Singh'
        over1=8
        totWick1=3
        overLastWick1=6
	'''

        '''		
        bowler1='Harbhajan Singh'
        batsman1='BJ Hodge'
        non_striker='SC Ganguly'
        over1=6
        totWick1=4
        overLastWick1=5
        '''

        predictions = {
            'plus1': Myrun(form.batsman.data, form.bowler.data, form.batsman_ns.data, form.over.data, form.totWick.data, form.overLastWick.data,1),
            'plus2': Myrun(form.batsman.data, form.bowler.data, form.batsman_ns.data, form.over.data, form.totWick.data, form.overLastWick.data,2),
            'plus3': Myrun(form.batsman.data, form.bowler.data, form.batsman_ns.data, form.over.data, form.totWick.data, form.overLastWick.data,3),               
            'plus4': Myrun(form.batsman.data, form.bowler.data, form.batsman_ns.data, form.over.data, form.totWick.data, form.overLastWick.data,4),
            #'plus1': Myrun(batsman1, bowler1, non_striker, over1,totWick1,overLastWick1,1),
            #'plus2': Myrun(batsman1, bowler1, non_striker, over1,totWick1,overLastWick1,2),
            #'plus3': Myrun(batsman1, bowler1, non_striker, over1,totWick1,overLastWick1,3),
            #'plus4': Myrun(batsman1, bowler1, non_striker, over1,totWick1,overLastWick1,4),
            }
        return render_template('predict.html', form=form, predictions=predictions)
    return render_template('predict.html', form=form, team1=team1, team2=team2)

# start server
if __name__ == '__main__':
    #print(batsman_list)
    #print(bowler_list)
    app.run(host= '127.0.0.1', port=5000, debug=True)

