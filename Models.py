from app import db

class Msg(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    msg = db.Column(db.String(), nullable=False)

    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    user = db.relationship('User',
        backref=db.backref('msgs', lazy=True))

    def __repr__(self):
        return self.msg


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=False, nullable=False)
    counter = db.Column(db.Integer, unique=False, nullable=False)
    facbook_id = db.Column(db.Integer, unique=False, nullable=False)

    def __repr__(self):
        return self.username


def create_user(username,counter,facbook_id):
    exist=User.query.filter_by(facbook_id=facbook_id).first()
    if exist:
        return exist
    else:
        user=User(username=username,counter=counter,facbook_id=facbook_id)
        db.session.add(user)
        db.session.commit()
        return user

def create_msg(msg,user_id):
    msg=Msg(msg=msg,user_id=user_id)
    db.session.add(msg)
    db.session.commit()
    return msg

def update_counter(counter,facbook_id):
    user = User.query.filter_by(facbook_id=facbook_id).first()
    user.counter=counter
    db.session.commit()
    return user

def get_counter(facbook_id):
    user = User.query.filter_by(facbook_id=facbook_id).first()
    db.session.commit()
    return user.counter

def get_user_name(facbook_id):
    user = User.query.filter_by(facbook_id=facbook_id).first()
    db.session.commit()
    return user.username

def get_user_msg(user_id):
    user_msg = Msg.query.filter_by(user_id=user_id).all()
    db.session.commit()
    return user_msg

