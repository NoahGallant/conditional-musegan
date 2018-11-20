from musegan.utils import midi_io
import pypianoroll


instruments = ['drums', 'guitar', 'piano', 'bass', 'strings']

type_inf = ['inference', 'interpolation']

nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

bort = ['fake_x_bernoulli_sampling','fake_x_hard_thresholding']

for instrument in instruments:
    for type_in in type_inf:
        for bt in bort:
            for num in nums:
                proll = pypianoroll.load('../exp/accompaniment/'+instrument+'/results/'+type_in+'/pianorolls/'+bt+'/'+bt+'_'+num+'.npz')
                pypianoroll.write(proll, '../MidiFiles/'+instrument+'/'+type_in+'/'+bt+'/'+bt+'_'+num+'.midi')





