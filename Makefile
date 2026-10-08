# Simulation of the RTL against the bit-exact Python reference.
#   make vectors   regenerate rtl/*_coefs.vh and tb/vectors/ from the model
#   make sim       run the self-checking testbench (Icarus Verilog)

RTL = rtl/hb_stage.v rtl/interp8.v rtl/pwm_corr.v rtl/ns_shaper.v \
      rtl/bd_pwm.v rtl/classd_channel.v

.PHONY: sim vectors model clean

sim: build/tb_classd
	vvp build/tb_classd

build/tb_classd: $(RTL) tb/tb_classd.v rtl/hb_coefs.vh rtl/ns_coefs.vh tb/vectors/counts.vh
	mkdir -p build
	iverilog -g2012 -Wall -I rtl -I tb/vectors -o $@ $(RTL) tb/tb_classd.v

vectors:
	python3 model/fixed_golden.py

model:
	python3 model/run_openloop.py
	python3 model/loop_design.py
	python3 model/run_closedloop.py

clean:
	rm -rf build

# ---- jitter_fifo ----
JF = jitter_fifo/rtl/jitter_fifo.v jitter_fifo/tb/tb_jitter_fifo.v

.PHONY: jf_sim jf_model
jf_sim:
	mkdir -p build
	iverilog -g2005 -Wall -o build/tb_jf_fast $(JF)
	iverilog -g2005 -Wall -Ptb_jitter_fifo.RD_PER=20.325 -o build/tb_jf_slow $(JF)
	vvp build/tb_jf_fast
	vvp build/tb_jf_slow

jf_model:
	python3 jitter_fifo/model/jfifo_model.py
	python3 jitter_fifo/model/stress.py
	python3 jitter_fifo/model/tb_scenario.py
