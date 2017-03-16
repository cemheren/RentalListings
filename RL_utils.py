import numpy as np
import pickle

# Import smtplib for the actual sending function
import smtplib

# Import the email modules we'll need
from email.mime.text import MIMEText




def prepare_submission_file(fname, result, ids_test):
    print '\n==> Preparing Submission File'
    submission_list = []
    submission_list.append("listing_id,high,medium,low")

    for i in range(len(result)):
        current_id = ids_test[i]
        r = result[i]
        line = str(current_id) + ',' + ",".join(map(str, r))
        submission_list.append(line)

    # Save Output of Testing
    sub_file = open(fname, 'w')
    for item in submission_list:
        sub_file.write("%s\n" % item)


# Todo: I plan to add a detailed report from history
def email_report(to_address):
    a = 1

# Integrate EMAIL LATER (ACTUALLY DONE)
# from_addr = 'tt.cmpeproject@gmail.com'
# to_addr = ['fermat4214@gmail.com']
# subject = 'Run Finished'
# body = 'All Done: Buraya Sistem Parametreleri ve History Dokulebilir'
#
# email_text = """\
# From: %s
# To: %s
# Subject: %s
#
# %s
# """ % (from_addr, ", ".join(to_addr), subject, body)
#
# server = smtplib.SMTP('smtp.gmail.com:587')
# server.ehlo()
# server.starttls()
# server.login('tt.cmpeproject','tt4214cmpe')
#
# server.sendmail('tt.cmpeproject@gmail.com', 'fermat4214@gmail.com', email_text)
# server.quit()



