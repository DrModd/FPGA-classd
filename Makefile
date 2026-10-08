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
JF_LS64 = -Ptb_jitter_fifo.AW=8 -Ptb_jitter_fifo.NT=64 -Ptb_jitter_fifo.MPH_SH=7 \
          -Ptb_jitter_fifo.DIV=128 -Ptb_jitter_fifo.TOL=40 \
          -Ptb_jitter_fifo.COEF_FILE='"jitter_fifo/rtl/fir_ls64_m128.hex"'

.PHONY: jf_sim jf_model
jf_sim:
	mkdir -p build
	# Catmull-Rom table (NT 4): bit-exact ramp check, -/+ 1000 ppm
	iverilog -g2005 -Wall -o build/tb_jf_cr4_fast $(JF)
	iverilog -g2005 -Wall -Ptb_jitter_fifo.RD_PER=20.325 -o build/tb_jf_cr4_slow $(JF)
	# 64-tap LS table (default of the module): AW 8, Fs = clk / 128, -/+ 3000 ppm
	iverilog -g2005 -Wall $(JF_LS64) -Ptb_jitter_fifo.RD_PER=20.406 -o build/tb_jf_ls64_fast $(JF)
	iverilog -g2005 -Wall $(JF_LS64) -Ptb_jitter_fifo.RD_PER=20.284 -o build/tb_jf_ls64_slow $(JF)
	vvp build/tb_jf_cr4_fast
	vvp build/tb_jf_cr4_slow
	vvp build/tb_jf_ls64_fast
	vvp build/tb_jf_ls64_slow

jf_model:
	python3 jitter_fifo/model/gen_coefs.py
	python3 jitter_fifo/model/jfifo_model.py
	python3 jitter_fifo/model/stress.py
	python3 jitter_fifo/model/tb_scenario.py
