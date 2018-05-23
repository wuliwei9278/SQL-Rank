# version 12.0:
# 1. Rewrite gradient calculation to reduce time complexity to O(d2_bar)
# 2. Remove line search and calculate average of gradients

# main("ml1m_full_tr.csv", "ml1m_full_te.csv", 0.1, 4, 100, 3)
# main("foursquare_oc_10_train_ratings.csv", "foursquare_oc_10_test_ratings.csv", 0.1, 4, 100, 3)
# main("usr_bookmarks_oc_50_train_ratings.csv", "usr_bookmarks_oc_50_test_ratings.csv",0.1, 4, 100, 3)
# main("checkin_gowalla.oc_train_ratingsALL.csv", "checkin_gowalla.oc_test_ratings100.csv", 0.1, 4, 100, 3)
# main("checkin_gowalla_oc_50_train_ratings.csv", "checkin_gowalla_oc_50_test_ratings.csv", 0.1, 4, 100, 3)
# main("ml1m_oc_50_train_ratings.csv", "ml1m_oc_50_test_ratings.csv", 0.1, 4, 100, 3)

function main(train, test, learning_rate, lambda, r, ratio)
	#train = "ml1m_oc_50_train_ratings.csv"
	#test = "ml1m_oc_50_test_ratings.csv"
	# requires ratio to be integer, usually 3 works best
	X = readdlm(train, ',' , Int64);
	x = vec(X[:,1]);
	y = vec(X[:,2]);
	v = vec(X[:,3]);
	Y = readdlm(test, ',' , Int64);
	xx = vec(Y[:,1]);
	yy = vec(Y[:,2]);
	vv = vec(Y[:,3]);
	n = max(maximum(x), maximum(xx)); 
	msize = max(maximum(y), maximum(yy));
	X = sparse(x, y, v, n, msize); # userid by movieid
	Y = sparse(xx, yy, vv, n, msize);
	# julia column major 
	# now moveid by userid
	X = X'; 
	Y = Y'; 
	rows = rowvals(X);
	vals = nonzeros(X);
	cols = zeros(Int, size(vals)[1]);
	index = zeros(Int, n + 1);

	d2, d1 = size(X);
	cc = 0;
	# need to record new_index based on original index
	# such that later, no need to shift each iteration, only need to swap the zero part
	new_len = 0;
	new_index = zeros(Int, d1 + 1);
	new_index[1] = 1;
	for i = 1:d1
		index[i] = cc + 1;
		tmp = nzrange(X, i);
		nowlen = size(tmp)[1];
		newlen = nowlen * (1 + ratio);
		new_len += newlen;
		new_index[i + 1] = new_index[i] + newlen;
		for j = 1:nowlen
			cc += 1;
			cols[cc] = i;
		end
	end
	index[d1 + 1] = cc + 1;
	# no need to sort for 0/1 data, so we don't need sort_input function in sqlrank10.jl 
	# ASSUMPTION: new_vals containing all 0's and 1's
	# we also need a function to shuffle all 1's and swapping 0's 
	new_rows = zeros(Int, new_len);
	new_cols = zeros(Int, new_len);
	new_vals = zeros(Int, new_len);
	for i = 1:d1
		rows_set = Set{Int}();
		for j = index[i]:(index[i + 1] - 1)
			push!(rows_set, rows[j]);
		end
		nowlen = new_index[i + 1] - new_index[i];
		nowOnes = div(nowlen, 1 + ratio);
		for j = 1:nowOnes
			new_rows[new_index[i] + j - 1] = rows[index[i] + j - 1];
			new_cols[new_index[i] + j - 1] = i;
			new_vals[new_index[i] + j - 1] = vals[index[i] + j - 1];
		end
		nowStart = new_index[i] + nowOnes;
		nowEnd = new_index[i + 1] - 1;
		for j = nowStart:nowEnd
			while true
				row_idx = rand(1:d2);
				if !(row_idx in rows_set)
					new_rows[j] = row_idx;
					new_cols[j] = i;
					new_vals[j] = 0.0;
					push!(rows_set, row_idx);
					break;
				end
			end  
		end
	end

	rows_t = rowvals(Y);
	vals_t = nonzeros(Y);
	cols_t = zeros(Int, size(vals_t)[1]);
	index_t =  zeros(Int, n + 1)
	cc = 0;
	for i = 1:d1
		index_t[i] = cc + 1;
		tmp = nzrange(Y, i);
		nowlen = size(tmp)[1];
		for j = 1:nowlen
			cc += 1
			cols_t[cc] = i
		end
	end
	index_t[d1 + 1] = cc + 1;

	# again, no need to sort rows_t, vals_t, cols_t under the ASSUMPTION
	# we also don't need levels
	
	srand(123456789);
	U = 0.1*randn(r, d1); 
	V = 0.1*randn(r, d2);
    #U = rand(r,d1)*(0.1 - (-0.1)) + (-0.1)
    #V = rand(r,d2)*(0.1 - (-0.1)) + (-0.1)
	m = comp_m(new_rows, new_cols, U, V);
	# no need to calculate max_d_bar, since we are using all 1's and 0's appended of ratio 1:ratio 
	println("rank: ", r, ", ratio of 0 vs 1: ", ratio, ", lambda:", lambda, ", learning_rate: ", learning_rate);
	# no need for obtain_R method, but we do need a shuffling method I call it stochasticQueuing 
	# (that is why I name the method as sqlrank: stochastic queuing listwise ranking algorithm)
	
	println("iter time objective_function precision@K = 1, 5, 10");
	obj = objective(new_index, m, new_rows, d1, lambda, U, V);
	p1,p2,p3=compute_precision(U, V, X, Y, d1, d2, rows, vals, rows_t, vals_t);
	println("[", 0, ",", obj, ", ", p1," ",p2," ",p3, "],");
    #println("[", 0, ",", obj, "],");

	totaltime = 0.00000;
	num_epoch = 121;
    num_iterations_per_epoch = 1;
	nowobj = obj;
	for epoch = 1:num_epoch
		tic();
		for iter = 1:num_iterations_per_epoch
			U, m = obtain_U(new_rows, new_cols, new_index, U, V, learning_rate, d1, r, lambda);
			V = obtain_V(new_rows, new_cols, new_index, m, U, V, learning_rate, d1, r, lambda);
		end
        
        new_rows = stochasticQueuing(new_rows, new_index, d1, d2, ratio);
        
        totaltime += toq();
	    #if (epoch - 1) % 3 == 0
        #    learning_rate = learning_rate * 0.3
        #end
        #learning_rate = learning_rate * 0.95
        if (epoch - 1) % 1 == 0
            learning_rate = learning_rate * 0.9
            p1,p2,p3=compute_precision(U, V, X, Y, d1, d2, rows, vals, rows_t, vals_t);
		    m = comp_m(new_rows, new_cols, U, V);
		    nowobj = objective(new_index, m, new_rows, d1, lambda, U, V);
		    println("[", epoch, ", ", totaltime, ", ", nowobj, ", ", p1,", ",p2,", ",p3, "],");
	    else
            m = comp_m(new_rows, new_cols, U, V);
            nowobj = objective(new_index, m, new_rows, d1, lambda, U, V);
            println("[", epoch, ", ", totaltime, ", ", nowobj);
        end
    end
end

function stochasticQueuing(rows, index, d1, d2, ratio)
	new_rows = zeros(Int, size(rows)[1]);
	for i = 1:d1
		nowlen = index[i + 1] - index[i];
		nowOnes = div(nowlen, 1 + ratio);
		newOrder = shuffle(1:nowOnes);
		rows_set = Set{Int}();
		for j = 1:nowOnes
			oldIdx = index[i] + j - 1;
			row_j = rows[oldIdx];
			push!(rows_set, row_j);
			newIdx = index[i] + newOrder[j] - 1;
			new_rows[newIdx] = row_j;
		end
		nowStart = index[i] + nowOnes;
		nowEnd = index[i + 1] - 1;
		for j = nowStart:nowEnd
			while true
				row_idx = rand(1:d2);
				if !(row_idx in rows_set)
					new_rows[j] = row_idx;
					push!(rows_set, row_idx);
					break;
				end
			end  
		end
	end
	return new_rows	
end

function obtain_U(rows, cols, index, U, V, s, d1, r, lambda)
	m = comp_m(rows, cols, U, V);
	grad_U = comp_gradient_U(rows, cols, index, m, U, V, s, d1, r, lambda);
	U = U - s * grad_U;
	m = comp_m(rows, cols, U, V);
	return U, m
end

function comp_gradient_U(rows, cols, index, m, U, V, s, d1, r, lambda)
	grad_U = zeros(size(U));
	for i = 1:d1
		d_bar = index[i+1] - index[i];
		grad_U[:,i] = comp_gradient_ui(rows, cols, index, d_bar, m, i, V, r);
	end
	grad_U += lambda * U;
	return grad_U
end

function comp_gradient_ui(rows, cols, index, d_bar, m, i, V, r)
	cc = zeros(d_bar);
	tt = 0.0;
	total = 0.0;
	for t = d_bar:-1:1
		tmp = m[index[i] - 1 + t];
		total += exp(tmp);
		tt += 1 / total;
	end
	total = 0.0;
	for t = d_bar:-1:1
		ttt = m[index[i] - 1 + t];
		cc[t] -= ttt * (1 - ttt);
		cc[t] += exp(ttt) * ttt * (1 - ttt) * tt;
		total += exp(ttt);
		tt -= 1 / total;
	end

	res = zeros(r);
	for t = 1:d_bar
		res += cc[t] * V[:,rows[index[i] - 1 + t]];
	end
	return res
end

function obtain_V(rows, cols, index, m, U, V, s, d1, r, lambda)
	grad_V = comp_gradient_V(rows, cols, index, m, U, V, s, d1, r, lambda);
	V = V - s * grad_V;
	return V
end

function comp_gradient_V(rows, cols, index, m, U, V, s, d1, r, lambda) 
	grad_V = zeros(size(V));
	for i = 1:d1
		d_bar = index[i+1] - index[i];
		cc = zeros(d_bar);
		tt = 0.0;
		total = 0.0;
		for t = d_bar:-1:1
			tmp = m[index[i] - 1 + t];
			total += exp(tmp);
			tt += 1 / total;
		end
		total = 0.0;
		for t = d_bar:-1:1
			ttt = m[index[i] - 1 + t];
			cc[t] -= ttt * (1 - ttt);
			cc[t] += exp(ttt) * ttt * (1 - ttt) * tt;
			total += exp(ttt);
			tt -= 1 / total;
		end

		for t = 1:d_bar
			j = rows[index[i] - 1 + t]
			grad_V[:,j] += cc[t] * U[:,i]
		end	
	end
	grad_V += lambda * V;
	return grad_V
end

function logit(x)
	return 1.0/(1+exp(-x))
end

function comp_m(rows, cols, U, V)
	m = zeros(length(rows));
	for i = 1:length(rows)
		m[i] = logit(dot(U[:,cols[i]], V[:,rows[i]]));
	end
	return m
end

function objective(index, m, rows, d1, lambda, U, V)
	res = 0.0;
	for i = 1:d1
		tt = 0.0;
		d_bar = index[i+1] - index[i];
		for t = d_bar:-1:1
			# since we will shuffle new_rows, new_cols, and obtain new m
			# we don't need to shuffle again for m (ASSUMPTION: we only have 1's and 0's)
			tmp = m[index[i] - 1 + t];
			tt += exp(m[index[i] - 1 + t]);
			res -= tmp;
			res += log(tt);
		end
	end
	res += lambda / 2 * (vecnorm(U) ^ 2 +vecnorm(V) ^ 2);
	return res
end

function compute_precision(U, V, X, Y, d1, d2, rows, vals, rows_t, vals_t)
	K = [1, 5, 10]; # K has to be increasing order
	precision = [0, 0, 0];
	for i = shuffle(1:d1)[1:1000]
	#for i = 1:d1
		tmp = nzrange(Y, i);
		test = Set{Int64}();
		for j in tmp
            push!(test, rows_t[j]);
		end
		#test = Set(rows_t[tmp])
		if isempty(test)
			continue
		end
		tmp = nzrange(X, i);
		vals_d2_bar = vals[tmp];
		train = Set(rows[tmp]);
		score = zeros(d2);
		ui = U[:, i];
		for j = 1:d2
			if j in train
				score[j] = -10e10;
				continue;
			end
			vj = V[:, j];
			score[j] = dot(ui,vj);
		end
		p = sortperm(score, rev = true);
		for c = 1: K[length(K)]
			j = p[c];
			if score[j] == -10e10
				break;
			end
			if j in test
				for k in length(K):-1:1
					if c <= K[k]
						precision[k] += 1;
					else
						break;
					end
				end
			end
		end
	end
	#precision = precision./K/d1;
	precision = precision./K/1000;
	return precision[1], precision[2], precision[3]
end
