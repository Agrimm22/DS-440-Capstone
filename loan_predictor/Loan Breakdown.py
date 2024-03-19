term=float(request.form['term'])
amnt=float(request.form['amnt'])/1000
rate= #predict based on new model and data
r=rate/12

MonthlyPayment=((amnt)*(r)*(1+r)**term)/(((1+r)**term)-1)
MonthlyPayment=round(MonthlyPayment, 2)
TotalInterestPaid=(MonthlyPayment*term)-amnt
TotalInterestPaid=round(TotalInterestPaid, 2)

#print("The interest for this loan will be: ", )
print("The total annual payment for this loan is:", MonthlyPayment*12)
print("The total monthly payment for this loan is:", MonthlyPayment)
print("The total interest paid for this loan is: ", TotalInterestPaid)


