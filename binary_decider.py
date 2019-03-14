import numpy as np
from sklearn.cluster import KMeans
from hdbscan import HDBSCAN

#Find and rank the Top N members from choices, based on a series of one v. one matchups.
#Works by assigning win/lose propabilities based on the user input and then propegating it
#through common opponents [if A>B and B>C, then A gets some points in A>C].
#Keeping a count of measurements for each matchup as they occur could help issues with triangling....
#i.e. self.cmatrix[a][b] += np.exp(-0.5*np.abs(val-1))
class BinaryChooser:
	def __init__(self):
		pass
	#Standard run method.
	#top selects how many choices to rank. A 0 means rank everything.
	#runs is something for stalling out
	#tol is a convergence criteria
	#goal selects how precise we want things before we finish
	#prob is the value assigned when we select a>b
	#prop_lim determines how far we propegate results of common opponents. A value of 2 or 3 is recommended.
	#tiers selects the tiers desired for outputs.
	def run(self,choices,top=0,goal=0.8,runs=10,tol=0.01,prob=0.8,prop_lim=4,tiers=None):
		self.labels = choices
		if top>0:
			self.limit = top
		else:
			self.limit = len(self.labels)
		self.sim_runs = 0
		self.goal=goal
		self.max_runs=runs
		self.tol=tol
		self.tiers=tiers
		self.base_prob=prob
		self.vals=[]
		self.prop_lim=prop_lim
		#Reset the internal score matrix 
		self.matrix = np.zeros((len(self.labels),len(self.labels)))
		while not self.check_convergence():
			#Randomly select a matchup
			a,b,ind_a,ind_b = self.get_choices()
			#Display/record the matchup from the user
			val = self.get_user_input(a,b)
			#Reset visited entries. Don't want a vs. a situations
			self.visited=[(a,a) for a in range(len(self.labels))]
			#Add results to the matrix
			self.propegate_result(-1*val,ind_a,ind_b)
		self.return_results(tiers=self.tiers)
		
	#A method to automatically run the decider using a list of numbers of the given length.
	#Intended to test convergence behavior or stress test for bugs.
	#vict_prob allows for probabilistic behavior (higher number wins vict_prob percentage of the time)
	#Setting a triangle index allows for a case where i>j>k>i.
	#I could add support for multiple triangles at a later date.
	def auto_run(self,length,top=0,goal=0.8,runs=10,tol=0.1,prob=0.8,prop_lim=4,vict_prob=1.0,triangle_index=None):
		#Most of this will be the same as the above run() method
		if top>0:
			self.limit=top
		else:
			self.limit = length
		#Our choice list is just going to be a list of numbers
		self.labels=np.arange(length)
		#unless we insert a triangle.
		#e.g., if triangle_index =20,
		#20 is replaced with 'i_20','j_20','k_20'.
		if triangle_index is not None and triangle_index < length:
			self.labels=list(self.labels)
			ind = self.labels.index(triangle_index)
			blah = self.labels.pop(ind)
			self.labels.append('i_%i' % triangle_index)
			self.labels.append('j_%i' % triangle_index)
			self.labels.append('k_%i' % triangle_index)
			self.labels=np.array(self.labels)
			if not top>0:
				self.limit+=2
		#Shuffle the input list to allow for better error testing
		np.random.shuffle(self.labels)
		self.sim_runs = 0
		self.goal=goal
		self.max_runs=runs
		self.tol=tol
		self.base_prob=prob
		self.vals=[]
		self.prop_lim=prop_lim
		self.matrix = np.zeros((len(self.labels),len(self.labels)))
		self.conv_steps=0
		while not self.check_convergence():
			self.conv_steps+=1
			a,b,ind_a,ind_b=self.get_choices()
			#Get the answer automatically instead of from a user
			val = self.get_auto_input(a,b,vict_prob=vict_prob)
			self.visited = [(a,a) for a in range(len(self.labels))]
			self.propegate_result(-1*val,ind_a,ind_b)
		return self.auto_return()
		
	#Automatically select the greater of a and b with probability vict_prob
	def get_auto_input(self,a,b,vict_prob=1.0):
		#Check if a triangle situation appears
		#If so, handle the values such that i>j>k>i.
		#Done by doing i=1,j=-1,k=2*opponent
		try:
			a = int(a)
		except:
			a = str(a)
		try:
			b = int(b)
		except:
			b= str(b)
			
		if type(a) == str and type(b)==str:
			if a.split('_')[0]=='i':
				a = 1
				if b.split('_')[0]=='j':
					b=-1
				else:
					b=2
			elif a.split('_')[0]=='j':
				a=-1
				if b.split('_')[0]=='i':
					b=1
				else:
					b=-2
			else:
				if b.split('_')[0]=='i':
					b=1
					a=2
				else:
					b=-1
					a=-2
		#If only one of them is a string replace it with it's attached number
		elif type(a)==str:
			a=int(a.split('_')[1])
		elif type(b)==str:
			b=int(b.split('_')[1])
		#Select the appropriate value and return -1 for left, 1 for right (dumb, I know)
		if a>b:
			if np.random.rand<=vict_prob:
				return 1
			else:
				return -1
		elif b>a:
			if np.random.rand<=vict_prob:
				return -1
			else:
				return 1
				
	#Return a set of convergence/accuracy information for the automatic testing			
	def auto_return(self):
		#Create the full matrix from the triangular matrix
		full_mat = self.matrix-self.matrix.T
		#Calculate all of the row scores. Higher is better.
		scores = np.sum(full_mat,axis=1)
		#print scores[list(self.labels).index(0)]
		#Collect the indices for sorting these scores from high to low, taking only the Top N
		score_args = np.argsort(scores)[::-1][:self.limit]
		labs=np.array(self.labels)[score_args]
		#Gulp. Some Top N stuff here. Only take the needed NxN matarix
		full_mat_ = full_mat[score_args][:,score_args]
		#print full_mat_.shape
		#Rescore everything
		scores = np.sum(full_mat_,axis=1)
		#print scores
		score_args = np.argsort(scores)[::-1]
		#labs are what the score args sort our input list to be
		labs = labs[score_args]
		labs_ = []
		for i in range(len(labs)):
			#print labs[i],type(labs[i])
			try:
				v_ = int(labs[i])
				labs_.append(v_)
			except:
				v_ =str(labs[i])
				labs_.append(int(v_.split('_')[1]))
		labs_=np.array(labs_)
		correct = np.sort(labs_)[::-1]
		#Sort the labels by what they should be....
		#Probably need to rewrite based for triangles.....
		#correct = np.sort(np.arange(len(self.labels)))[::-1][:self.limit]
		scores=scores[score_args]
		#Check to see if everything matches from the correct order and the returned order
		correct_ = np.all(correct==labs_)
		#Calculate the percent of the array predicted correctly
		corr_per = np.nonzero(correct==labs_)[0].shape[0]/float(len(correct))
		#Compute the RMSE for the computed order)
		corr_scr = np.sqrt(np.mean((correct-labs_)**2))
		#A second metric doing favorable MU's - unfavorable MU's. Works poorly.
		matchups =[np.nonzero(full_mat_[i]>=self.goal)[0].shape[0]-np.nonzero(full_mat_[i]<=(-1*self.goal))[0].shape[0] for i in score_args]
		mu_args = np.argsort(matchups)[::-1]
		mu_scores_ = np.array(matchups)[mu_args]
		#print self.labels
		#print correct
		#print labs[mu_args]
		correct_2 = np.all(correct==labs_[mu_args])
		corr_per2 = np.nonzero(correct==labs_[mu_args])[0].shape[0]/float(len(correct))
		corr_scr2 = np.sqrt(np.mean((correct-labs_[mu_args])**2))
		#Third metric where the favorable and unfavorable MU's are then weighted
		#by how good/bad the opponent was.
		mu_scores2=[]
		#Norm the worst score to one
		bump = -1 * np.min(scores) + 1
		#Norms the best score to 1
		fall = np.max(scores)+1
		#calc the running total
		for i in range(full_mat_.shape[0]):
			running=0
			for per in (full_mat_[score_args])[i]:
				if per>=self.goal:
					running+= (scores[i]+bump)
				elif per<=-1*self.goal:
					running+=(scores[i]-fall)
			mu_scores2.append(running)
		mu_args2 = np.argsort(mu_scores2)[::-1]
		mu_scores2_ = np.array(mu_scores2)[mu_args2]
		correct_3 = np.all(correct==labs_[mu_args2])
		corr_per3 = np.nonzero(correct==labs_[mu_args2])[0].shape[0]/float(len(correct))
		corr_scr3 = np.sqrt(np.mean((correct-labs_[mu_args2])**2))
		
		return self.conv_steps,((len(self.labels)**2-len(self.labels))/2),zip(labs,scores,matchups,mu_scores2),correct_,corr_per,corr_scr,correct_2,corr_per2,corr_scr2,correct_3,corr_per3,corr_scr3
		
	#Check to determine if we have a converged set of results	
	def check_convergence(self):
		#tvals = self.matrix[np.tril_indices(len(self.labels),k=-1)]
		#val = np.sort(np.abs(tvals))[-1*min(self.limit,l]
		#Compute the full matrix from the triangular one
		full_mat = self.matrix-self.matrix.T
		#Same as score_args from above
		entry_sort = np.argsort(np.sum(full_mat,axis=1))[::-1]
		entries = entry_sort[:self.limit]
		#Grab the NxN matrix
		full_mat= full_mat[entries][:,entries]
		#val = np.min(np.abs(full_mat))
		#Only get one set of the symmetric values
		tvals = np.abs(full_mat[np.tril_indices(full_mat.shape[0],k=-1)])
		#print len(tvals)
		#If every matchup is accurate, we're done
		if np.min(tvals)>=self.goal:
			print 'Condition 1'
			return True
		#If everything averages to something better than self.goal, we're done
		if np.mean(tvals)>= (1+self.goal)/2.:
			return True
		#tvals = np.abs(full_mat[np.tril_indices(full_mat.shape[0],k=-1)])
		#Now we're worried about ending the script if it's never converging.
		tvals = np.sort(tvals)
		if tvals[0]>0.001:
			#tvals_ = tvals[:len(tvals)/10]
			val = np.mean(tvals)
			if len(self.vals)>= self.max_runs:
				self.vals=self.vals[1:]+[val]
			else:
				self.vals.append(val)
			if len(self.vals)==self.max_runs:
				#If the average convergence hasn't changed much in max_runs, end the script
				#(we also have to properly scale the convergence criteria, self.tol)
				if np.max(self.vals)-np.min(self.vals)<(self.tol/self.limit):#self.matrix.shape[0]):
					print 'Condition 2'
					return True 
				mid = self.max_runs/2
				#Split into two halves, in case of more chaotic behavior
				if np.abs(np.max(self.vals[:mid])-np.max(self.vals[mid:]))<(self.tol/(2.*self.limit)):#self.matrix.shape[0])):
					print 'Condition 3'
					return True 
		return False	
	
	#Randomaly select the choices presented to the user.
	#Prioritze lower certainty matchups, matchups with no data, and matchups within the top N
	def get_choices(self):
		full_mat=self.matrix-self.matrix.T
		row_sums = np.sum(np.abs(full_mat),axis=1)
		score_args = np.argsort(np.sum(full_mat,axis=1))[::-1][:self.limit]
		#Use an exponential distribution based on amount of MU data
		weights = np.exp(-2*row_sums)
		#Prioritize labels in the top N
		wt = len(self.labels)/float(self.limit)
		for sa in score_args:
			weights[sa]*=(4*wt)
		weights/=np.sum(weights)
		#Get the first opponent
		ind_a = np.random.choice(np.arange(row_sums.shape[0]),p=weights)
		#Get the second opponent
		col_vals = np.abs(full_mat[ind_a,:])
		weights = np.exp(-2*col_vals)
		#Prevent a vs. a matchups
		weights[ind_a]=0.0
		for sa in score_args:
			weights[sa]*=(4*wt)
		#Upweight zero info matchups
		for i,w in enumerate(weights):
			if w>0.9999 and w<1.00001:
				weights[i]*=2*wt*wt
		weights/=np.sum(weights)
		ind_b = np.random.choice(np.arange(col_vals.shape[0]),p=weights)
		return self.labels[ind_a],self.labels[ind_b],ind_a,ind_b
		
	#Present the matchup to the user and record their input
	def get_user_input(self,a,b):
		v_= 'r'
		#Only except a 0 or 1 as input. All other inputs should raise an error.
		while v_ not in ["0","1"]:
			v_ = raw_input('Who is stronger? %s or %s? Enter "0" to select the first choice and "1" to select the second choice.' % (a,b))
			if v_ not in ["0","1"]:
				print 'Please enter either "0" or "1".'
		if v_=="0":
			return -1
		elif v_=="1":
			return 1
			
	#The crux of the entire class.
	#The idea is to take a triangular matchup matrix (triangular, as a vs. b = -1*b vs. a), self.matrix,
	#and update it when a winner is selected. -1 = a is certain to lose and 1 = a is certain to win.
	#For directly selected matchups, we increase/decrease by self.base_prob, with diminishing returns.
	#These are given by by doing new = old + (win/loss) * (1-(old*win/loss))*base_prob
	#Next, we propegate this result, so if a>b and b>c, then a>c gets some points,
	#with decays based on exp(-0.5*steps) and certainty of b>c (stored in mult).
	#Positive val means a wins, negative means b wins.
	def propegate_result(self,val,ind_a,ind_b,base=None,check=None,mult=1):
		#Handle things differently for the first pass.
		#Could be it's own method.....
		if np.abs(val)==1:
			#Matrix entries to check for propegaton
			self.to_visit=[]
			#print ind_a,ind_b,type(ind_a)
			#Transform results to keep the matrix triangular (lower left)
			if ind_a>ind_b:
				val_ = self.matrix[ind_a][ind_b]
				#Update the primary value
				val__ = val_ + np.sign(val)*((1-(np.sign(val)*val_))*self.base_prob)
				self.matrix[ind_a,ind_b]=val__
				self.visited.append((ind_a,ind_b))
			else:
				val_ = self.matrix[ind_b][ind_a]
				val__ = val_ + -1*np.sign(val)*((1-(np.sign(-1*val)*val_))*self.base_prob)
				self.matrix[ind_b,ind_a]=val__
				self.visited.append((ind_b,ind_a))
			#print val,ind_a,ind_b
			#Propegate the next step, looking at all of the matchups a and b have taken part in.
			#Has a sign swith. If a>b, we want b>c cases (so b wins instead of loses).
			for i in range(self.matrix.shape[0]):
				self.to_visit.append((2*val,ind_b,i,ind_a,i,1))
				self.to_visit.append((-2*val,ind_a,i,ind_b,i,1))
			#print len(self.to_visit)
			#print val_,val__,val
			#print self.matrix
			#Better than recursion and prevents out of order visits. self.to_visit will grow as necessary.
			while len(self.to_visit)>0:
				params = self.to_visit.pop(0)
				#Having the VERY occasional, unexplained error here 
				try:
					#Look for a>b, c<b cases, so that we can update a vs. c's value in self.matrix
					self.propegate_result(params[0],params[1],params[2],base=params[3],check=params[4],mult=params[5])
				except:
					print params, self.to_visit,val,ind_a,ind_b,base,check,mult
					raise 
		#Deal with all of the propegation cases. 
		else:
			#print 'Propegate 1'
			og_val = val
			#If we are on the wrong half of the matrix, we need to temporarily reverse things. 
			if ind_b>ind_a:
				#self.propegate_result(-1*val,ind_b,ind_a,base=base,check=check,mult=mult)
				val= og_val*-1
				a_=ind_a
				b_ = ind_b
				ind_a = b_
				ind_b = a_
			#else:
			v_=self.matrix[ind_a][ind_b]
			#Do not update if no matchup has happened, we have already propegated a result here, we are in a a>b, c>b situation,
			#we have hit the propegation limit, or we are updating a vs. a
			if v_==0 or (ind_a,ind_b) in self.visited or np.sign(val)!=np.sign(v_) or np.abs(val)>self.prop_lim or check==base:
				#pass
				#print 'Nevermind', v_, (ind_a,ind_b),val
				#if v_!=0:
					#print (self.visited)
				return
			else:
				#Get the appropriate value to update
				if base>check:
					base_v = self.matrix[base][check]
				else:
					base_v = self.matrix[check][base]
					#val*=-1
			#If we are looking at (b,c) to update (c,a), we need to reverse signs.
			if max(base,check)!=ind_a and min(base,check)!=ind_b:
				#print ind_a, ind_b, max(base,check),min(base,check)
				v_*=-1
				val*=-1
			#Update the matrix
			v_ = base_v+  (1-(base_v*np.sign(val)))*v_*np.exp(-0.5*(np.abs(val)-1))*mult
			if base>check:
				#print 'Update', self.matrix[ind_a][ind_b],v_, val,ind_a,ind_b,base,check
				self.matrix[base][check]=v_
				#print self.matrix
				#print self.matrix
			else:
				#val*=-1
				#print 'Update', self.matrix[ind_a][ind_b],v_,val,ind_a,ind_b,base,check
				self.matrix[check][base]=v_
				#print self.matrix
			self.visited.append((ind_a,ind_b))
			#print og_val,val,base,check
			#Appropriately search for new propegation targets.
			if np.abs(og_val)+1 <= self.prop_lim:
				for i in range(self.matrix.shape[0]):
					#if check==ind_b:
					self.to_visit.append((np.sign(og_val)*(np.abs(val)+1),check,i,base,i,mult*np.abs(self.matrix[ind_a][ind_b])))
					#elif check==ind_a:
						#self.to_visit.append((-1*np.sign(val)*(np.abs(val)+1),check,i,base,i,mult*np.abs(self.matrix[ind_a][ind_b])))
		
	#The method for returning results to the user.
	#Returns a list broken into an appropropriate set of tiers.
	#I have arbitrarily set a limit of 9 tiers.
	def return_results(self,tiers):
		#Get the full matrix from the triangular matrix, get the row sums, and sort by them
		full_mat = self.matrix-self.matrix.T
		scores = np.sum(full_mat,axis=1)
		score_args = np.argsort(scores)[::-1][:self.limit]
		labs = list(np.array(self.labels)[score_args])
		scores=scores[score_args]
		tier_names = ['God','S','A','B','C','D','E','F','Garbage']
		max_tiers=9
		#If the user has not specified a tier number to use, use HDBSCAN to come up with the number
		if tiers is None:
			tiers=0
		if tiers<=0:
			clusterer= HDBSCAN(min_cluster_size=2,min_samples=1,metric='l1')
			clusterer.fit(scores.reshape(-1,1))
			tiers=max(clusterer.labels_)
			#print clusterer.labels_
		tiers = min(max_tiers,tiers+1)
		#Use KMeans to split the scores into the set number of tiers
		clusterer=KMeans(n_clusters=tiers)
		try:
			clusterer.fit(scores.reshape(-1,1))
		except OverflowError:
			print scores,tiers
			raise
		l_pairs = zip(list(scores),list(clusterer.labels_),labs)
		curr_tier = -1
		#Create the return string.
		for scr,tier,lab in l_pairs:
			if tier != curr_tier:
				print '\n'
				print tier_names.pop(0)+':'
			curr_tier=tier
			print '%s %.2f' % (lab,scr)
			
	def get_prob_stats(self,keys,runs=100):
		return_dict={}
		if type(keys[0])==str:
			keys_=[eval(k) for k in keys]
			keys=keys_
		for key in keys:
			steps=[]
			correct1_scrs=[]
			correct1_pers=[]
			correct2_scrs=[]
			correct2_pers=[]
			correct3_scrs=[]
			correct3_pers=[]
			return_dict[str(key)]={}
			for i in range(runs):
				if len(key)==3:
					stuff = self.auto_run(key[0],prop_lim=key[1],goal=key[2])
				elif len(key)==4:
					stuff = self.auto_run(key[0],prop_lim=key[1],goal=key[2],vict_prob=key[3])
				elif len(key)==5:
					stuff = self.auto_run(key[0],prop_lim=key[1],goal=key[2],vict_prob=key[3],top=key[4])
				steps.append(stuff[0])
				correct1_scrs.append(stuff[5])
				correct1_pers.append(stuff[4])
				correct2_pers.append(stuff[7])
				correct2_scrs.append(stuff[8])
				correct3_pers.append(stuff[10])
				correct3_scrs.append(stuff[11])
			return_dict[str(key)]['scores']=[np.mean(steps),np.min(steps),np.max(steps),np.std(steps)]
			return_dict[str(key)]['correct1']=[np.mean(correct1_pers),np.std(correct1_pers),np.mean(correct1_scrs),np.std(correct1_scrs)]
			return_dict[str(key)]['correct2']=[np.mean(correct2_pers),np.std(correct2_pers),np.mean(correct2_scrs),np.std(correct2_scrs)]
			return_dict[str(key)]['correct3']=[np.mean(correct3_pers),np.std(correct3_pers),np.mean(correct3_scrs),np.std(correct3_scrs)]
		return return_dict
	
		