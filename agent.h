/**
 * Framework for Threes! and its variants (C++ 11)
 * agent.h: Define the behavior of variants of agents including players and environments
 *
 * Author: Theory of Computer Games
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include <fstream>
#include "board.h"
#include "action.h"
#include "weight.h"

class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss("name=unknown role=unknown " + args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			meta[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag = "") {}
	virtual action take_action(const board& b) { return action(); }
	virtual bool check_for_win(const board& b) { return false; }

public:
	virtual std::string property(const std::string& key) const { return meta.at(key); }
	virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }
	virtual std::string name() const { return property("name"); }
	virtual std::string role() const { return property("role"); }

protected:
	typedef std::string key;
	struct value {
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
};

/**
 * base agent for agents with randomness
 */
class random_agent : public agent {
public:
	random_agent(const std::string& args = "") : agent(args) {
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};

/**
 * base agent for agents with weight tables and a learning rate
 */
class weight_agent : public agent {
public:
	weight_agent(const std::string &args = "") : agent(args), alpha(0.0125), opcode({0, 1, 2, 3})
	{
		if (meta.find("init") != meta.end())
			init_weights(meta["init"]);
		if (meta.find("load") != meta.end())
			load_weights(meta["load"]);
		if (meta.find("alpha") != meta.end())
			alpha = float(meta["alpha"]);
	}
	virtual ~weight_agent() {
		if (meta.find("save") != meta.end())
			save_weights(meta["save"]);
	}
	virtual action take_action(const board& before){
		board::reward op_best = -1;
		board::grid best_state;
		float r_V_best = -1.0;
		    
		for (int op : opcode)
		{
			board tmp = before;
			board::reward reward = tmp.slide(op);
			if(reward == -1){
				continue;
			}
			board::reward Fplun = tmp.value();
			board::reward F = before.value();
			board::reward Reward = Fplun - F;
			board::grid state = board(before).state_plun(op);
			std::vector<int> tuples = get_tuple(state); 
			float V = get_V(tuples);
			float r_V = Reward + V;
			if (r_V > r_V_best)
			{
				r_V_best = r_V;
				best_state = state;
				op_best = op;
			}
		}

		if(train){
			update_weight(r_V_best);
		}
		best_state = board(before).state_plun(op_best);
		s_tuples = get_tuple(best_state);
		s_V = get_V(s_tuples);

		train=1;
		 
		if (op_best != -1)
			return action::slide(op_best);
		return action();
	} 
	void update_weight(const float st_1_r_V){
		for (int i = 0; i < 8; i++)
		{
			net[i][s_tuples[i]] = net[i][s_tuples[i]] + alpha * (st_1_r_V - s_V);
		}
	}
	void last_update(){
		for (int i = 0; i < 8; i++)
		{
			net[i][s_tuples[i]] = net[i][s_tuples[i]] + alpha * (0 - s_V);
		}
	}
	float get_V(const std::vector<int>tuples){
		 float result=0;
		 for(int i=0;i<8;i++){
			 result += net[i][tuples[i]];
		 }
		 return result;
	 }
	std::vector<int> get_tuple(const board::grid state){
		 std::vector<int> tuples;
		 std::vector<int> col1;
		 std::vector<int> col2;
		 std::vector<int> col3;
		 std::vector<int> col4;
		 for(int i=0;i<4;i++){
			 board::row row = state[i];
			 int a = row[0];
			 col1.emplace_back(a);
			 int b = row[1];
			 col2.emplace_back(b);
			 int c = row[2];
			 col3.emplace_back(c);
			 int d = row[3];
			 col4.emplace_back(d);
			 tuples.emplace_back(16*16*16*a + 16*16*b + 16*c + d);
		 }
		 tuples.emplace_back(16*16*16*col1[0] + 16*16*col1[1] + 16*col1[2] + col1[3]);
		 tuples.emplace_back(16*16*16*col2[0] + 16*16*col2[1] + 16*col2[2] + col2[3]);
		 tuples.emplace_back(16*16*16*col3[0] + 16*16*col3[1] + 16*col3[2] + col3[3]);
		 tuples.emplace_back(16*16*16*col4[0] + 16*16*col4[1] + 16*col4[2] + col4[3]);
		 return tuples;
	 }
 

protected:
	virtual void init_weights(const std::string& info) {
		/*std::string res = info; // comma-separated sizes, e.g., "65536,65536"
		for (char& ch : res)
			if (!std::isdigit(ch)) ch = ' ';
		std::stringstream in(res);
		for (size_t size; in >> size; net.emplace_back(size));*/
		for(int i=0;i<8;i++){
			net.emplace_back(65536);
		}
	}
	virtual void load_weights(const std::string& path) {
		std::ifstream in(path, std::ios::in | std::ios::binary);
		if (!in.is_open()) std::exit(-1);
		uint32_t size;
		in.read(reinterpret_cast<char*>(&size), sizeof(size));
		net.resize(size);
		for (weight& w : net) in >> w;
		in.close();
	}
	virtual void save_weights(const std::string& path) {
		std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
		if (!out.is_open()) std::exit(-1);
		uint32_t size = net.size();
		out.write(reinterpret_cast<char*>(&size), sizeof(size));
		for (weight& w : net) out << w;
		out.close();
	}

protected:
	std::vector<weight> net;
	float alpha = 0.0125;
	std::array<int, 4> opcode;
	float s_V;
	bool train = 0;
	std::vector<int> s_tuples;
};

/**
 * default random environment, i.e., placer
 * place the hint tile and decide a new hint tile
 */
class random_placer : public random_agent {
public:
	random_placer(const std::string& args = "") : random_agent("name=place role=placer " + args) {
		spaces[0] = { 12, 13, 14, 15 };
		spaces[1] = { 0, 4, 8, 12 };
		spaces[2] = { 0, 1, 2, 3};
		spaces[3] = { 3, 7, 11, 15 };
		spaces[4] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
	}

	virtual action take_action(const board& after) {
		std::vector<int> space = spaces[after.last()];
		std::shuffle(space.begin(), space.end(), engine);
		for (int pos : space) {
			if (after(pos) != 0) continue;

			int bag[3], num = 0;
			for (board::cell t = 1; t <= 3; t++)
				for (size_t i = 0; i < after.bag(t); i++)
					bag[num++] = t;
			std::shuffle(bag, bag + num, engine);

			board::cell tile = after.hint() ?: bag[--num];
			board::cell hint = bag[--num];

			return action::place(pos, tile, hint);
		}
		return action();
	}

private:
	std::vector<int> spaces[5];
};

/**
 * random player, i.e., slider
 * select a legal action randomly
 */
class random_slider : public random_agent {
public:
	random_slider(const std::string& args = "") : random_agent("name=slide role=slider " + args),
		opcode({ 0, 1, 2, 3 }) {}

	virtual action take_action(const board& before) {
		std::shuffle(opcode.begin(), opcode.end(), engine);
		for (int op : opcode) {
			board::reward reward = board(before).slide(op);
			if (reward != -1) return action::slide(op);
		}
		return action();
	}

private:
	std::array<int, 4> opcode;
};

class my_slider : public agent {
	public:
		my_slider(const std::string &args = "") : agent(args), opcode({0, 1, 2, 3}) {
			if (meta.find("init") != meta.end())
				std::cout << "This is the args(fuck): " << args << std::endl;
		}
		virtual ~my_slider()
		{
			if (meta.find("save") != meta.end())
				std::cout << "This is the save(fuck): " << std::endl;
		}

		virtual action take_action(const board& before){
			board:: reward reward_best = -1;
			board:: reward op_best = -1;
			for (int op : opcode){
				board::reward Reward = board(before).slide(op);
				std::cout<< "reward:" << Reward << std::endl;
				if (Reward > reward_best){
					reward_best = Reward;
					op_best = op;
				}
			}
			if (op_best != -1)
				return action::slide(op_best);
			return action();
		}
	private:
		std::array<int, 4> opcode;
};