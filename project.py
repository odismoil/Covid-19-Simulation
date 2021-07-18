import numpy as np
from numpy.random import randn, randint, random
import matplotlib.pyplot as plt

### Part 1: wrtiting functions that would be necessary for simulations

def system_initialize(num_infect, num_asymptomatic, 
                      public_places = False):
    '''
    this function creates dataset to initialize the system. The form of 
    this dataset Nx8 if public_places = False, Nx10 if public_places= True
    
    dataset: [pos, v, Status, I_s, t_illness, I_SS] 
    pos = x,y coordinates of people
    v = vx,vy of people (random normal)
    Status: 0 = Susceptible; 1 = Infected; 2 = Recovered; 3:Dead (SIRD)
    I_s = 1 (binary) means show symptoms/ go to see doctor when infected 
    I_s = 0 (binary) means does not show symptoms/ careless with respect to illness
    t_illness = time passed since getting infected
    I_SS 1: (binary) after incubation time passed, show symptoms and go
    to isolation (needed for isolation part only)
    I_SS 0: (binary) does not show signs or incubation time not passed or
    careless who do not go to see doctors
    
    '''

    pos = np.mgrid[:450.0:32j, :450.0:32j].reshape(2, -1).T #positions based on square lattice
    v = np.empty([0,2])
    while len(v)<len(pos):
        rand_num = randn(1,2)
        vel = 0.8*rand_num
        v_person = (vel[0,0]**2 + vel[0,1]**2)**0.5
        if v_person<1.12: # v should be less than 1.12 m/s (see report)
            v = np.append(v,vel, axis = 0)
    
    v[:,0] = v[:,0] - v[:,0].mean()
    v[:,1] = v[:,1] - v[:,1].mean()
    Status = np.zeros((len(pos),1))
    rand_num = randint(0,len(pos),num_infect)
    Status[rand_num] = 1
    
    I_s = np.ones((len(pos),1))
    rand_num = randint(0,len(pos),num_asymptomatic)
    I_s[rand_num] = 0
    
    t_illness = np.zeros((len(pos),1))
    
    I_SS = np.zeros((len(pos),1)) 
    
    
    if public_places == True:
        public = np.zeros((len(pos),1))
        t_public = np.zeros((len(pos),1))
        dataset = np.concatenate((pos, v, Status, I_s, t_illness, I_SS, public, t_public), axis = 1)
        '''
        dataset: [pos, v, Status, I_s, t_illness, I_SS, public, t_public] 
        in addition to above description:
            public: (binary) 1: in public 0: not in public
            t_public: time spent in public place
        '''
    else:
        dataset = np.concatenate((pos, v, Status, I_s, t_illness, I_SS), axis = 1)
        
    return dataset

def get_infection(dataset, d, P_infection):
    '''this function finds new people who get infected by being close to
    the other infected people within distance d
    '''
    for i in range(len(dataset)):
        if dataset[i,4] == 1 or dataset[i,4] == 2 or dataset[i,4] == 3:
            continue # 1: already infected 2: recovered 3: dead
        
        for j in range(len(dataset)):
            r_ij = ((dataset[i,0]-dataset[j,0])**2 + (dataset[i,1]-dataset[j,1])**2)**0.5
            
            if i==j or r_ij > d:
                continue
            
            if dataset[j,4] == 1:
                p_i = random()
                if p_i <= P_infection:
                    dataset[i,4] = 1


def outcome(dataset, p_death, T_illness, dead_count):
    '''
    this function calculates outcome of infected people after illness
    passed. status 2 means recovered, while 3 means dead

    '''
    for i in range(len(dataset)):
        if dataset[i,6] >= T_illness:
            if dataset[i,4] == 1:
                p_i = random()
                if p_i <= p_death:
                    dataset[i,4] = 3
                    dead_count += 1
                else:
                    dataset[i,4] = 2  
                    
    return dead_count


def social_distance(dataset):
    '''
    this function calculates sosio - psychological force which is in 
    Coulombic form. Forces at r_ij > cut_off are  very small,
    so they are neglected

    '''
    k =10; q1 = 2; q2 = 2; cut_off = 25; F_X = []; F_Y = []
    for i in range(len(dataset)):
        f_x = 0; f_y = 0
        for j in range(len(dataset)):
            r_ij = ((dataset[i,0]-dataset[j,0])**2 + (dataset[i,1]-dataset[j,1])**2)**0.5
            
            if i==j or r_ij > cut_off:
                continue
            
            F_mag = k*q1*q2/r_ij**3
            f_x += F_mag*(dataset[i,0]-dataset[j,0])
            f_y += F_mag*(dataset[i,1]-dataset[j,1])
            
        F_X.append(f_x)
        F_Y.append(f_y)
        
    return np.array(F_X), np.array(F_Y)


def isolation(dataset, isolated, T_illness, T_incubation, dead_count):
    '''
    This function takes people who are infected and show symptoms
    to isolation (out of town or saparate dataset (isolation).
     When infection passed they are either recovered or
    dead. Recovered people are returned to the town again

    '''
    #isolation part
    for i in range(len(dataset)):
        if dataset[i,4] == 1:
            if dataset[i,5] == 1:
                if dataset[i,6] >= T_incubation:
                    dataset[i,7] = 1
                    
    isolated = np.append(isolated, dataset[dataset[:,7] == 1], axis = 0)
    dataset = np.delete(dataset, dataset[:,7] == 1, axis = 0)
    
    
    # de-isolation part
    
    p_death = 0.05;
    dead_count = outcome(isolated, p_death, T_illness, dead_count)
    dead_count = outcome(dataset, p_death, T_illness, dead_count)
    
    dataset = np.append(dataset, isolated[isolated[:,4] ==2], axis = 0)
    isolated = np.delete(isolated, isolated[:,4] == 2, axis = 0)
        
    return dataset, isolated, dead_count


def public_place(dataset, v_original, x, y, p_public, dt, T_public):
    '''
    this function takes people randomly and brings them to public places
    such as universities, school or any mass-gatherings. Brought people's 
    speed reduced 10 times. After passage of T_public time (average time
    spent at public), their original speeds are returned and they move away
    from this place quickly.

    '''
    for i in range(len(dataset)):
        p = random()
        if p <= p_public and dataset[i,8] == 0:
            dataset[i,8] = 1
            dataset[i,2:4] = 0.1*v_original[i,:]
            dataset[i,0] = x; dataset[i,1] = y
            
        if dataset[i,9] >= T_public:
            dataset[i,2:4] = v_original[i,:]
            dataset[i,8] = 0
            dataset[i,9] = 0
            
    dataset[dataset[:,8] == 1, 9] += dt
    

def boundary_cond(dataset, x, y):
    ''' hard wall boundary condition'''
    
    dataset[dataset[:,0] < 0, 2] = -1*dataset[dataset[:,0] < 0, 2]
    dataset[dataset[:,0] < 0, 0] = -1*dataset[dataset[:,0] < 0, 0]

    dataset[dataset[:,0] > x, 2] = -1*dataset[dataset[:,0] > x, 2]
    dataset[dataset[:,0] > x, 0] = 2*x - dataset[dataset[:,0] > x, 0]
     

    dataset[dataset[:,1] < 0, 3] = -1*dataset[dataset[:,1] < 0, 3]
    dataset[dataset[:,1] < 0, 1] = -1*dataset[dataset[:,1] < 0, 1]

    dataset[dataset[:,1] > y, 3] = -1*dataset[dataset[:,1] > y, 3]
    dataset[dataset[:,1] > y, 1] = 2*y - dataset[dataset[:,1] > y, 1]
    
                      



### Part 2: Simulation of various scenarios
dt = 1; d = 2; T_illness = 250; T_incubation = 100; p_infect = 0.4
p_death = 0.05;  m = 70


##case 1: no restriction
dataset = system_initialize(10, 200)
dead_count = 0; active_cases_1 = []

for i in range(1800):
    get_infection(dataset,d,p_infect) #Susceptible people get infected
    dataset[dataset[:,4] == 1, 6] += dt #t_illness counts time of illness for infected people only
    dead_count = outcome(dataset, p_death, T_illness, dead_count) #fate of ill people: rocovery or death
    
    dataset[:,0] = dataset[:,0] + dataset[:,2]*dt # people's movement in city
    dataset[:,1] = dataset[:,1] + dataset[:,3]*dt
    
    boundary_cond(dataset, 450, 450) #hard-wall boundary conditions
    
    if i%20 == 0: #counting # of active people in some specific timeframe
        count = (dataset[:,4] == 1).sum()
        active_cases_1.append(count)

total_infected_1 = 1024 - (dataset[:,4] == 0).sum()
dead_count_1 = dead_count
print("number of dead people: ", dead_count, 'total_infected: ', total_infected_1)



## case 2: introduction of hygiene
dataset = system_initialize(10, 200)
dead_count = 0; active_cases_2 = []; p_infect = 0.2  #probability of 
#getting infected is reduced by half, reset necessary values

for i in range(1800):
    get_infection(dataset,d,p_infect)
    dataset[dataset[:,4] == 1, 6] += dt
    dead_count = outcome(dataset, p_death, T_illness, dead_count)
    
    dataset[:,0] = dataset[:,0] + dataset[:,2]*dt
    dataset[:,1] = dataset[:,1] + dataset[:,3]*dt
    
    boundary_cond(dataset, 450, 450)
    
    if i%20 == 0:
        count = (dataset[:,4] == 1).sum()
        active_cases_2.append(count)

total_infected_2 = 1024 - (dataset[:,4] == 0).sum()
dead_count_2 = dead_count
print("number of dead people: ", dead_count, 'total_infected: ', total_infected_2)



##case 3: isolation of infected
dataset = system_initialize(10, 200)
isolated = np.empty((0,8))
dead_count = 0; p_infect = 0.4; active_cases_3 = [] #reset necessary values

for i in range(1800):    
    get_infection(dataset,d,p_infect)
    # take infected people who show signs to isolation
    dataset, isolated, dead_count = isolation(dataset, isolated, T_illness, T_incubation, dead_count)
    dataset[dataset[:,4] == 1, 6] += dt
    
    if len(isolated) > 0:
        isolated[:,6] += dt

    dataset[:,0] = dataset[:,0] + dataset[:,2]*dt
    dataset[:,1] = dataset[:,1] + dataset[:,3]*dt
    
    boundary_cond(dataset, 450, 450)
    
    if i%20 == 0:
        count = (dataset[:,4] == 1).sum()
        active_cases_3.append(count)

total_infected_3 = 1024 - (dataset[:,4] == 0).sum()
dead_count_3 = dead_count
print("number of dead people: ", dead_count, 'total_infected: ', total_infected_3)



##case 4: isolation of infected + number of asymptomatic/careless people increased 
dataset = system_initialize(10, 500)
isolated = np.empty((0,8))
dead_count = 0; p_infect = 0.4; active_cases_4 = [] #reset necessary values

for i in range(1800):    
    get_infection(dataset,d,p_infect)
    dataset, isolated, dead_count = isolation(dataset, isolated, T_illness, T_incubation, dead_count)
    dataset[dataset[:,4] == 1, 6] += dt
    
    if len(isolated) > 0:
        isolated[:,6] += dt

    dataset[:,0] = dataset[:,0] + dataset[:,2]*dt
    dataset[:,1] = dataset[:,1] + dataset[:,3]*dt
    
    boundary_cond(dataset, 450, 450)
    
    if i%20 == 0:
        count = (dataset[:,4] == 1).sum()
        active_cases_4.append(count)

total_infected_4 = 1024 - (dataset[:,4] == 0).sum()
dead_count_4 = dead_count
print("number of dead people: ", dead_count, 'total_infected: ', total_infected_4)
    


##case 5: introduction of public_places
dataset = system_initialize(10, 200, public_places= True)
v = np.copy(dataset[:,2:4])
dead_count = 0; T_public = 7; p_public = 0.005; active_cases_5 = [] #reset necessary values
x0 = 225; y0 = 225;

for i in range(1800):
    get_infection(dataset,d,p_infect)
    dataset[dataset[:,4] == 1, 6] += dt
    dead_count = outcome(dataset, p_death, T_illness, dead_count)
    #modeling public places
    public_place(dataset, v, x0, y0, p_public, dt, T_public)
    
    dataset[:,0] = dataset[:,0] + dataset[:,2]*dt
    dataset[:,1] = dataset[:,1] + dataset[:,3]*dt
    
    boundary_cond(dataset, 450, 450)
    
    if i%20 == 0:
        count = (dataset[:,4] == 1).sum()
        active_cases_5.append(count)
 
total_infected_5 = 1024 - (dataset[:,4] == 0).sum()
dead_count_5 = dead_count
print("number of dead people: ", dead_count, 'total_infected: ', total_infected_5)



##case 6: social distance
dataset = system_initialize(10, 200)
dt = 0.3; dead_count = 0; active_cases_6 = [] #reset necessary values

for i in range(6000):
    
    get_infection(dataset,d,p_infect)
    dataset[dataset[:,4] == 1, 6] += dt
    dead_count = outcome(dataset, p_death, T_illness, dead_count)  
    # modeling social-distancing
    F_X, F_Y = social_distance(dataset)
    
    if i>0: #verlet algorithm is implemented in this part
        dataset[:,2] = dataset[:,2] + F_X*dt/(2*m)
        dataset[:,3] = dataset[:,3] + F_Y*dt/(2*m)

    dataset[:,0] = dataset[:,0] + dataset[:,2]*dt + 0.5*F_X*dt**2/m
    dataset[:,1] = dataset[:,1] + dataset[:,3]*dt + 0.5*F_Y*dt**2/m
    
    dataset[:,2] = dataset[:,2] + F_X*dt/(2*m)
    dataset[:,3] = dataset[:,3] + F_Y*dt/(2*m)

    boundary_cond(dataset, 450, 450)
    
    if i%67 == 0:
        count = (dataset[:,4] == 1).sum()
        active_cases_6.append(count)
    
total_infected_6 = 1024 - (dataset[:,4] == 0).sum()
dead_count_6 = dead_count
print("number of dead people: ", dead_count, 'total_infected: ', total_infected_6)  

##case 7: 30% of population vaccinated
dataset = system_initialize(10, 200)
dead_count = 0; dt = 1; active_cases_7 = [] #reset necessary values
infect = np.array((dataset[:,4] == 1).nonzero()) # position of infected people in dataset
vaccinated = []

while len(vaccinated) < 308: #308 people are vaccinated
    rand_number = randint(0,1024)
    #vaccinated people should be other than infected people, and we also drop if same random number appears more than once 
    if (rand_number == infect).any() or (rand_number == np.array(vaccinated)).any():
        continue
    else:
        vaccinated.append(rand_number)
        
dataset[vaccinated, 4] = 2 # vaccinated people

for i in range(1800):
    get_infection(dataset,d,p_infect)
    dataset[dataset[:,4] == 1, 6] += dt
    dead_count = outcome(dataset, p_death, T_illness, dead_count)
    
    dataset[:,0] = dataset[:,0] + dataset[:,2]*dt
    dataset[:,1] = dataset[:,1] + dataset[:,3]*dt
    
    boundary_cond(dataset, 450, 450)
    
    if i%20 == 0:
        count = (dataset[:,4] == 1).sum()
        active_cases_7.append(count)

total_infected_7 = 1024 - (dataset[:,4] == 0).sum() - 308
dead_count_7 = dead_count
print("number of dead people: ", dead_count, 'total_infected: ', total_infected_7)



#graph of active cases vs time
Time = np.linspace(0, 1800, 90)
plt.plot(Time, active_cases_1, '--', color = 'k')
plt.plot(Time, active_cases_2, '-', color = 'b')
plt.plot(Time, active_cases_3, '-', color = 'r')
plt.plot(Time, active_cases_4, '-.', color = 'c')
plt.plot(Time, active_cases_5, '--', color = 'g')
plt.plot(Time, active_cases_6[:90], '*', color = 'y')
plt.plot(Time, active_cases_7, '.',color = 'm')
plt.legend(['Sc 1', 'Sc 2', 'Sc 3', 'Sc 4', 'Sc 5', 'Sc 6', 'Sc 7'])
plt.xlabel("time (s)")
plt.ylabel('active cases')
plt.title("Active Cases")
plt.show()

#writing out active cases data vs time
file = open('active_cases.txt', 'w')
print('time    Sc 1    Sc 2    Sc 3    Sc 4' '\t'
      'Sc 5     Sc 6    Sc 7', file = file)
for i in range(90):
    print(int(Time[i]), '\t', active_cases_1[i],'\t', active_cases_2[i], '\t', 
          active_cases_3[i], '\t', active_cases_4[i], '\t', active_cases_5[i], '\t', 
          active_cases_6[i], '\t', active_cases_7[i], file = file)
file.close()