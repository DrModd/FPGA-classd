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
