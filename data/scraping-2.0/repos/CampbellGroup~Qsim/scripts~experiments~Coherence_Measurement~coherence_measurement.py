import labrad
from Qsim.scripts.pulse_sequences.coherence_measurement_point import CoherenceMeasurementPoint as sequence
from Qsim.scripts.experiments.qsimexperiment import QsimExperiment
from labrad.units import WithUnit as U
import numpy as np


class CoherenceMeasurement(QsimExperiment):
    """
    Scan delay time between microwave pulses with variable pulse area
    """

    name = 'Coherence Measurement'

    exp_parameters = []
    exp_parameters.append(('MicrowaveInterrogation', 'AC_line_trigger'))
    exp_parameters.append(('MicrowaveInterrogation', 'delay_from_line_trigger'))
    exp_parameters.append(('Modes', 'state_detection_mode'))
    exp_parameters.append(('ShelvingStateDetection', 'repetitions'))
    exp_parameters.append(('StandardStateDetection', 'repetitions'))
    exp_parameters.append(('StandardStateDetection', 'points_per_histogram'))
    exp_parameters.append(('StandardStateDetection', 'state_readout_threshold'))
    exp_parameters.append(('Shelving_Doppler_Cooling', 'doppler_counts_threshold'))

    exp_parameters.extend(sequence.all_required_parameters())

    exp_parameters.remove(('EmptySequence', 'duration'))

    def initialize(self, cxn, context, ident):
        self.ident = ident

    def run(self, cxn, context):

        if self.p.MicrowaveInterrogation.AC_line_trigger == 'On':
            self.pulser.line_trigger_state(True)
            self.pulser.line_trigger_duration(self.p.MicrowaveInterrogation.delay_from_line_trigger)

        scan_parameter = self.p.MicrowaveRamsey.scan_type
        mode = self.p.Modes.state_detection_mode
        if mode == 'Shelving':
            self.setup_coherence_shelving_datavault()
        self.setup_datavault('time', 'probability')  # gives the x and y names to Data Vault
        self.setup_grapher('Microwave Ramsey Experiment')
        self.dark_time = self.get_scan_list(self.p.CoherenceMeasurement.delay_times, 'ms')
        for i, dark_time in enumerate(self.dark_time):
            should_break = self.update_progress(i/float(len(self.dark_time)))
            if should_break:
                break
            self.p['EmptySequence.duration'] = U(dark_time, 'ms')
            self.program_pulser(sequence)
            if mode == 'Shelving':
                [doppler_counts, detection_counts] = self.run_sequence(max_runs=500, num=2)
                self.dv.add(np.column_stack((np.arange(len(doppler_counts)), np.array(detection_counts), np.array(doppler_counts))), context=self.counts_context)
                errors = np.where(doppler_counts <= self.p.Shelving_Doppler_Cooling.doppler_counts_threshold)
                counts = np.delete(detection_counts, errors)
            else:
                [counts] = self.run_sequence()
            if i % self.p.StandardStateDetection.points_per_histogram == 0:
                hist = self.process_data(counts)
                self.plot_hist(hist)
            pop = self.get_pop(counts)
            self.dv.add(dark_time, pop)

    def setup_coherence_shelving_datavault(self):
        # datavault setup for the run number vs probability plots
        self.counts_context = self.dv.context()
        self.dv.cd(['coherence_measurement', 'shelving_counts'], True, context=self.counts_context)

        self.coherence_counts_dataset = self.dv.new('counts', [('run', 'arb')],
                                                    [('counts', 'detection_counts', 'num'), ('counts', 'doppler_counts', 'num')],
                                                    context=self.counts_context)

        for parameter in self.p:
            self.dv.add_parameter(parameter, self.p[parameter], context=self.counts_context)

    def finalize(self, cxn, context):
        self.pulser.line_trigger_state(False)
        self.pulser.line_trigger_duration(U(0.0, 'us'))
        pass


if __name__ == '__main__':
    cxn = labrad.connect()
    scanner = cxn.scriptscanner
    exprt = CoherenceMeasurement(cxn=cxn)
    ident = scanner.register_external_launch(exprt.name)
    exprt.execute(ident)
