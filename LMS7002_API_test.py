from __future__ import print_function
# import numpy
from matplotlib.pyplot import *
from pyLMS7002M import *
import ctypes
from ctypes import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack


def logTxt(text, end="\n"):
    print(text, end=end)
    sys.stdout.flush()
    
limeSuiteDll = ctypes.WinDLL (".\\LimeSuite.dll")

p1 = create_string_buffer('\000' * 1024)
result = limeSuiteDll.LMS_GetDeviceList(p1)

print ("result = " + str(result))
print ("parameter = " + repr(p1.value))

device = c_void_p()

##################################################################
# Option 1: Connect directly
# if(limeSuiteDll.LMS_Open(ctypes.byref (device), None, None) != 0):
    # print ("LimeSDR Connection Failed")
##################################################################
# Option 2: Connect via LimeSDR, then this object can be used in the script...
logTxt("Searching for LimeSDR... ",end="")
try:
    limeSDR = LimeSDR(usbBackend = "LimeAPI")
except:
    logTxt("\nLimeSDR not found")
    exit(1)
logTxt("FOUND")    
device = limeSDR.cyDev.getDevice()
##################################################################

# if (limeSuiteDll.LMS_Init(device) != 0):
    # print ("Error: Could not initialize device")
    
LMS_CH_RX = c_bool(0)

# if (limeSuiteDll.LMS_EnableChannel(device, LMS_CH_RX, 0, c_bool(1)) != 0):
    # print ("Error: Could not enable channel")
    
# LO_freq = c_double(800e6)

# print ("LO frequency = " + str(LO_freq.value) + " Hz")

# # Set center frequency to 800 MHz
# # Automatically selects antenna port
# if (limeSuiteDll.LMS_SetLOFrequency(device, LMS_CH_RX, 0, LO_freq) != 0):
    # print ("Error: Could not set LO frequency")
    
# if (limeSuiteDll.LMS_GetLOFrequency(device, LMS_CH_RX, 0, ctypes.byref(LO_freq)) != 0):
    # print ("Error: Could not get LO frequency")
# else:
    # print ("LO frequency = " + str(LO_freq.value) + " Hz")
    
# # Set sample rate to 8 MHz, ask to use 2x oversampling in RF
# # This set sampling rate for all channels
# sampleRate = 8.0e6
# if (limeSuiteDll.LMS_SetSampleRate(device, c_double(sampleRate), 2) != 0):
    # print ("Error: Could not set sampling rate")

### Here there is an unexpected behaviour unless the sample rate is previously set by the LMS_SetSampleRate function
# # host_Hz     sampling rate used for data exchange with the host
# host_Hz = c_double()
# # rf_Hz       RF sampling rate in Hz
# rf_Hz = c_double()
    
# if (limeSuiteDll.LMS_GetSampleRate(device, LMS_CH_RX, 0, ctypes.byref(host_Hz), ctypes.byref(rf_Hz)) != 0):
    # print ("Error: Could not get sampling rate")
# else:    
    # print ("Sampling rate used for data exchange with the host = " + str(host_Hz.value) + " Hz")
    # print ("RF sampling rate in Hz = " + str(rf_Hz.value) + " Hz")
    # sampleRate = host_Hz.value

# LMS_TESTSIG_NCODIV8 = 1
    
# # Enable test signal generation
# # To receive data from RF, remove this line or change signal to LMS_TESTSIG_NONE
# if (limeSuiteDll.LMS_SetTestSignal(device, LMS_CH_RX, 0, LMS_TESTSIG_NCODIV8, 0, 0) != 0):
    # print ("Error: Could not set test signal")
    
class lms_stream_t(Structure):
    _fields_ = [("handle", c_size_t),
                ("isTx", c_bool),
                # ("channel", c_uint32_t),
                ("channel", c_uint32),
                # ("fifoSize", c_uint32_t),
                ("fifoSize", c_uint32),
                ("throughputVsLatency", c_float),
                ("dataFmt", c_int)]

LMS_FMT_F32 = 0
LMS_FMT_I16 = 1
LMS_FMT_I12 = 2
                
streamId = lms_stream_t() #stream structure
streamId.channel = 0; #channel number
streamId.fifoSize = 1024 * 128; #fifo size in samples
streamId.throughputVsLatency = 1.0; #optimize for max throughput
streamId.isTx = LMS_CH_RX; #RX channel
# streamId.dataFmt = LMS_FMT_I16; #16-bit integers
streamId.dataFmt = LMS_FMT_I12; #16-bit integers
if (limeSuiteDll.LMS_SetupStream(device, ctypes.byref(streamId)) != 0):
    print ("Error: Could not set up stream")

# Initialize data buffers
# buffersize_python = 4096
# buffersize_python = 5000
# buffersize_python = 8192
buffersize_python = 16834
# buffersize_python = 2*16834
bufersize = c_int(buffersize_python) #complex samples per buffer
buffersize_int16 = 2 * buffersize_python
buffer = (c_int16 * buffersize_int16)() #buffer to hold complex values (2*samples))

# Start streaming
if (limeSuiteDll.LMS_StartStream(ctypes.byref(streamId)) != 0):
    print ("Error: Could not start stream")
    
samplesRead = c_int()
# Receive samples
samplesRead = limeSuiteDll.LMS_RecvStream(ctypes.byref(streamId), pointer(buffer), bufersize, None, 1000)
print ("Received " + str(samplesRead) + " samples")

I_samples = []
Q_samples = []

# TEMP, TEMP, TEMP, TEMP, TEMP
# samplesRead = 100

for j in range(0, samplesRead):
    I_samples.append(buffer[2*j])
    Q_samples.append(buffer[2*j + 1])
    
y_I = numpy.array(I_samples)
y_Q = numpy.array(Q_samples)

# sampleNumber = numpy.array(range(0, samplesRead))
# plot(sampleNumber, I_samples_array)
# plot(sampleNumber, Q_samples_array)
# legend(['I', 'Q'], loc='upper left')
# xlabel('Sample Number')
# ylabel('Samples')
# show()

# Number of samplepoints
N = samplesRead
# Sample spacing
T = 1.0 / sampleRate

yf_I = scipy.fftpack.fft(y_I)
yf_Q = scipy.fftpack.fft(y_Q)
xf_IQ = np.linspace(0, samplesRead-1, samplesRead)*sampleRate/samplesRead - sampleRate/2
xf_IQ_MHz = xf_IQ / 1E6

y_IQ = y_I + 1j*y_Q
yf_IQ = scipy.fftpack.fft(y_IQ)
yf_IQ = scipy.fftpack.fftshift(yf_IQ)

yf_IQ_mod = []
for i in range(0, samplesRead):
    currIQ = math.sqrt(yf_IQ.real[i]**2 + yf_IQ.imag[i]**2)
    currIQ /= samplesRead
    # The following line is needed if the data format is LMS_FMT_I16. If the format is LMS_FMT_I12 it should be commented out.
    # currIQ /= 2**4 - 1
    yf_IQ_mod.append(currIQ)
    
yf_IQ_dBFS = []
for i in range(0, samplesRead):
    if yf_IQ_mod[i] != 0:
        yf_IQ_dBFS.append(20 * np.log10(yf_IQ_mod[i]) - 69.2369)
    else:
        yf_IQ_dBFS.append(-300)
        
plot(xf_IQ_MHz, yf_IQ_dBFS)
xlabel('Frequency [MHz]')
ylabel('Amplitude [dBFS]')
axes = gca()
axes.set_ylim([-100,0])
plt.show()

xmarker = 1E6
ymarker = np.interp(xmarker, xf_IQ, yf_IQ_dBFS)
print("Marker (f= " +str(xmarker/1E6)+ " MHz) = " + str(ymarker) + " dBFS")

xmarker = -1E6
ymarker = np.interp(xmarker, xf_IQ, yf_IQ_dBFS)
print("Marker (f= " +str(xmarker/1E6)+ " MHz) = " + str(ymarker) + " dBFS")

ymarker = max(yf_IQ_dBFS)
xmarker_index = yf_IQ_dBFS.index(ymarker)
xmarker = xf_IQ[xmarker_index]
print("Peak Marker (f= " +str(xmarker/1E6)+ " MHz) = " + str(ymarker) + " dBFS")