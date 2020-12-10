using POMDPs, POMDPModelTools, POMDPSimulators, BasicPOMCP

struct NursingHomePOMDP <: POMDP{NTuple{6,Int64}, Symbol, NTuple{2,Int64}}
    num_residents::Int64 # total number of residents; must equal num_low_risk + num_high_risk
    num_low_risk::Int64 # number of low-risk individuals (default 5)
    num_high_risk::Int64 # number of high-risk individuals (default 5)
    r_lockdown::Int64 # reward for lockdown (default -150)
    r_healthy::Int64 # reward for each healthy person when not on lockdown (default 15)
    r_infected::Int64 # reward for each infected person when not on lockdown (default -25)
    r_recovered::Int64 # reward for each recovered person when not on lockdown (default 5)
    r_dead::Int64 # reward for each deceased person when not on lockdown (default -200)
    p_positive_healthy::Float64 # probability of a healthy individual testing positive (default .02)
    p_positive_infected::Float64 # probability of an infected individual testing positive (default .95)
    discount_factor::Float64 # discount
end

NursingHomePOMDP() = NursingHomePOMDP(10, 5, 5, -150, 15, -25, 5, -200, 0.02, 0.95, 0.95)

# The state of the nursing home POMDP is a tuple containing the number of residents who are, respectively: healthy/low-risk, healthy/high-risk, infected/low-risk, infected/high-risk, recovered, dead.
POMDPs.states(pomdp::NursingHomePOMDP) = reshape(collect(Iterators.product(0:pomdp.num_low_risk, 0:pomdp.num_high_risk, 0:pomdp.num_low_risk, 0:pomdp.num_high_risk, 0:pomdp.num_residents, 0:pomdp.num_residents)), (1, (pomdp.num_low_risk+1)^2 * (pomdp.num_high_risk+1)^2 * (pomdp.num_residents+1)^2))

POMDPs.stateindex(pomdp::NursingHomePOMDP, s::NTuple{6,Int64}) = s[1]*10^10 + s[2]*10^8 + s[3]*10^6 + s[4]*10^4 + s[5]*10^2 + s[6]+1
POMDPs.statetype(pomdp::NursingHomePOMDP) = NTuple{6,Int64}

# Two possible actions for the nursing home: go into lockdown or be open
POMDPs.actions(pomdp::NursingHomePOMDP) = [:lockdown, :open]
function POMDPs.actionindex(pomdp::NursingHomePOMDP, a::Symbol)
    if a == :lockdown
        return 1
    elseif a == :open
        return 2
    end
    error("invalid NursingHomePOMDP action: $a")
end;

function POMDPs.transition(pomdp::NursingHomePOMDP, s::Tuple{6,Int64}, a::Symbol)
    s0, s1, s2, s3, s4, s5 = s
    # If nursing home decides to lock down, then nobody becomes infected.
    if a == "lockdown"
	sp2 = 0
	sp3 = 0
	sp4 = s4
	sp5 = s5
	# determine next state for each low-risk infected person
	for i in 1:s2
	    rand = rand()
	    # 25% chance of recovering within the week
	    if rand < .25
		sp4 += 1
		# 2% chance of dying within the week
	    elseif rand > .98
		sp5 += 1
	    else
		sp2 += 1
	    end
	end
	# determine next state for each high-risk infected person
	for i in 1:s3
	    rand = rand()
	    # 17% chance of recovering within the week
	    if rand < .17
		sp4 += 1
		# 3% chance of dying within the week
	    elseif rand > .97
		sp5 += 1
	    else
		sp3 += 1
	    end
	end
	println("Old state: [$s0, $s1, $sp2, $sp3, $sp4, $sp5]; Action: $a; New state: [$s0, $s1, $sp2, $sp3, $sp4, $sp5]")
	return (s0, s1, sp2, sp3, sp4, sp5)::NTuple{6,Int64}
    else # if the nursing home decided to open
	sp0 = 0
	sp1 = 0
	sp2 = 0
	sp3 = 0
	sp4 = s4
	sp5 = s5
	# determine next state for each low-risk healthy person
	for i in 1:s0
	    rand = rand()
	    # 15% chance of becoming infected
	    if rand < .15
		sp2 += 1
	    else
		sp0 += 1
	    end
	end
	# determine next state for each high-risk healthy person
	for i in 1:s1
	    rand = rand()
	    # 15% chance of becoming infected
	    if rand < .15
		sp3 += 1
	    else
		sp1 += 1
	    end
	end
	# determine next state for each low-risk infected person
	for i in 1:s2
	    rand = rand()
	    # 25% chance of recovering within the week
	    if rand < .35
		sp4 += 1
	    # 1.5% chance of dying within the week
	    elseif rand > .95
		sp5 += 1
	    else
		sp2 += 1
	    end
	end
	# determine next state for each high-risk infected person
	for i in 1:s3
	    rand = rand()
	    # 20% chance of recovering within the week
	    if rand < .25
		sp4 += 1
	    # 4% chance of dying within the week
	    elseif rand > .95
		sp5 += 1
	    else
		sp3 += 1
	    end
	end
	println("Old state: [$s0, $s1, $sp2, $sp3, $sp4, $sp5]; Action: $a; New state: [$s0, $s1, $sp2, $sp3, $sp4, $sp5]")
	return (sp0, sp1, sp2, sp3, sp4, sp5)::NTuple{6,Int64}
    end
end;

# Observation is a tuple containing the number of positive tests that week for low-risk and high-risk residents, respectively
POMDPs.observations(pomdp::NursingHomePOMDP) = reshape(collect(Iterators.product(0:pomdp.num_low_risk, 0:pomdp.num_high_risk)), (1, (pomdp.num_low_risk+1) * (pomdp.num_high_risk+1)))
POMDPs.obsindex(pomdp::NursingHomePOMDP, o::Tuple{2,Int64}) = o[1]*10^2 + o[2]

function POMDPs.observation(pomdp::NursingHomePOMDP, a::Symbol, sp::Tuple{6,Int64})
    sp0, sp1, sp2, sp3, sp4, sp5 = sp
    pos_low = 0
    # Each healthy person has a 2% chance of testing positive
    for i in 1:sp0
	rand = rand()
	if rand < pomdp.p_positive_healthy
	    pos_low += 1
	end
    end
    # Each infected person has a 95% chance of testing positive
    for i in 1:sp2
	rand = rand()
	if rand < pomdp.p_positive_infected
	    pos_low += 1
	end
    end
    
    pos_high = 0
    # Repeat the process to determine test results of high-risk individuals
    for i in 1:sp1
	rand = rand()
	if rand < pomdp.p_positive_healthy
	    pos_high += 1
	end
    end
    for i in 1:sp3
	rand = rand()
	if rand < pomdp.p_positive_infected
	    pos_high += 1
	end
    end
    
    return (pos_low, pos_high)
end

function POMDPs.reward(pomdp::NursingHomePOMDP, s::Tuple{6,Int64}, a::Symbol, sp::Tuple{6,Int64})
    if a == "lockdown"
	# println("Reward: -150")
	return pomdp.r_lockdown
    else
	sp0, sp1, sp2, sp3, sp4, sp5 = sp
	r = pomdp.r_healthy*(sp0+sp1) - pomdp.r_infected*(sp2+sp3) + pomdp.r_recovered*sp4 - pomdp.r_dead*sp5
	# println("Reward: $r")
	return r
    end
end

POMDPs.initialstate(pomdp::NursingHomePOMDP) = (pomdp.num_low_risk, pomdp.num_high_risk, 0, 0, 0, 0)
POMDPs.discount(pomdp::NursingHomePOMDP) = pomdp.discount_factor


# Run the simulation

m = NursingHomePOMDP()
solver = POMCPSolver(max_time = 0.1)
policy = solve(solver, m)

rsum = 0.0
for (s,b,a,o,r) in stepthrough(m, policy, "s,b,a,o,r", max_steps=10)
    println("s: $s, b: $([pdf(b,s) for s in states(m)]), a: $a, o: $o")
    global rsum += r
end
println("Undiscounted reward was $rsum.")

