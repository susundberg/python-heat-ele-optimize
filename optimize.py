import csv
import pylab
import logging
import argparse
LOG = logging.getLogger("root")
SKIP_ROWS = 1 # 4*30*24
LIMIT_ROWS = SKIP_ROWS + 2**24 

CREATE_PLOTS = False

def load_prices():
    prices = []
    first_date = None
    last_date =None
    with open('chart.csv', newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for loop,row in enumerate(csv_reader):
            if loop == 0:
                continue
            if loop == 1:
                first_date = row[0]
            if loop < SKIP_ROWS:
               continue
            if loop >= LIMIT_ROWS:
               break 
            prices.append(float(row[1]))
            last_date = row[0]
    LOG.info("Loaded %d rows - between %s -> %s", len(prices), first_date, last_date)
    return prices

def load_temp():
    temps = []
    first_date = None
    last_date =None
    with open('temp.csv', newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for loop,row in enumerate(csv_reader):
            if loop == 0:
                continue
            if loop == 1:
                first_date = " ".join( row[1:5] )
            if loop < SKIP_ROWS:
               continue

            if loop >= LIMIT_ROWS:
               break 

            try:
                temps.append(float(row[5]))
            except ValueError:
                LOG.info("Cannot convert temp: %s", row)
                temps.append( temps[-1] )
            last_date = " ".join( row[1:5] )
    LOG.info("Loaded %d rows - between %s -> %s", len(temps), first_date, last_date)
    return temps




class TooLowTemp(Exception):
    pass


MACHINE_POWER_KW = 8
MACHINE_COP_1 = (35, 4.7)
MACHINE_COP_2 = (45, 3.7)
MACHINE_OCCUPIED_C = -25
HEAT_DROP_M10C_HOURS = 6
HEAT_DROP_M10C_OUTSIDE_C = -20
MACHINE_HEAT_C = 1
MACHINE_HEAT_OUTSIDE_C = -20
ROOM_TEMPERATURE_C=25

ADJUST_TH_HIGH = 55
ADJUST_TH_LOW = 40
ADJUST_TH_OFF = 45

def water_equivalent( lit ):
    # energy_for_model_1c = 
    water_1000l_1c = 1.1666676
    # lit =  / water_1000l_1c
    # lit * water_1000l_1c = (1.0/self.capasity_c_p_kwh)
    # self.capasity_c_p_kwh = 1.0 / (lit * water_1000l_1c)
    return 1.0 / ( lit/1000.0 * water_1000l_1c )
        
class Model:
    def __init__(self):
        
        self.machine_heat_kwh = MACHINE_POWER_KW 
        self.heat_loss_kwh_p_c = 0.2 # self.machine_heat_kwh  / (ROOM_TEMPERATURE_C - MACHINE_OCCUPIED_C)
        self.capasity_c_p_kwh = water_equivalent(360+270) # MACHINE_HEAT_C / ( self.machine_heat_kwh - self.heat_loss_kwh_p_c * (ROOM_TEMPERATURE_C - MACHINE_HEAT_OUTSIDE_C) )
        
        self.normal_on_c = ADJUST_TH_LOW
        self.normal_off_c = ADJUST_TH_OFF
        # self.save_th_low_c = 20
        self.save_th_high_c = ADJUST_TH_HIGH
        self.state_temp = self.normal_on_c

    def increase_power(self, amoutn):
        
        self.machine_heat_kwh *= amoutn

    def calculate_day( self, start_temp, day_mask, prices, temperatures ):
        loop_temp_c = start_temp
        usage_mask = [0.0, ] * 24
        system_temp = [0, ] * 24
        sum_price = 0
        sum_cons_dir = 0
        sum_cons_mach = 0
        assert( len(prices) == 24 )
        assert( len(temperatures) == 24 )
        assert( len(day_mask) == 24 )

        for hloop in range(0,24):
            heat_loss_kwh = (+25 - temperatures[hloop]) * self.heat_loss_kwh_p_c
            if heat_loss_kwh < 0:
                heat_loss_kwh = 0
            old_temp = loop_temp_c
            loop_temp_c = loop_temp_c - heat_loss_kwh*self.capasity_c_p_kwh 
            #if hloop == 12:
            #    LOG.info("Heat loss: %0.2f my temp: %0.2f -- cap %f", heat_loss_kwh, loop_temp_c, self.capasity_c_p_kwh  )
            if day_mask[hloop] == 1 and loop_temp_c < self.normal_on_c:     
                usage, temp_d = self.calculate_turn_on( self.normal_off_c - loop_temp_c)
                loop_temp_c += temp_d
                usage_mask[hloop] = usage
            
            elif day_mask[hloop] == 2:
                usage, temp_d = self.calculate_turn_on( self.save_th_high_c - loop_temp_c)
                loop_temp_c += temp_d
                usage_mask[hloop] = usage

            dir_ele_cons = 0
            if loop_temp_c < self.normal_on_c:
                # Direct electricity heating
                c_diff = (self.normal_on_c - loop_temp_c)
                dir_ele_cons = (1.0/self.capasity_c_p_kwh)*c_diff
                loop_temp_c += c_diff

            price = max( prices[hloop], 0.01 )
            mach_ele_cons = 0
            if usage_mask[hloop] > 0:
                mach_ele_cons = usage_mask[hloop] * self.calculate_ele_cons( 0.5*(old_temp + loop_temp_c) )
                
            sum_price += (mach_ele_cons + dir_ele_cons)*price
            system_temp[hloop] = loop_temp_c
            sum_cons_dir  += dir_ele_cons
            sum_cons_mach += mach_ele_cons
        
        return ( sum_price / 100.0, (usage_mask, system_temp, sum_cons_mach, sum_cons_dir ))

    def calculate_turn_on( self, tdiff_c ): 
        kw_required = (1.0/self.capasity_c_p_kwh)*tdiff_c
        kw_used = min( kw_required, self.machine_heat_kwh )

        temp_gained = kw_used*self.capasity_c_p_kwh 
        return (kw_used / self.machine_heat_kwh, temp_gained)

    def calculate_ele_cons( self, temp ):
        
        # defined at two 
        cop = 0
        if temp < MACHINE_COP_1[0]:
            cop = MACHINE_COP_1[1]
        cop_d = MACHINE_COP_2[1] - MACHINE_COP_1[1]
        temp_d = MACHINE_COP_2[0] - MACHINE_COP_1[0]
        temp_v = (temp - MACHINE_COP_1[0])/temp_d
        cop = temp_v * cop_d + MACHINE_COP_1[1]
        #LOG.info("CALCULATE COP %f %f", temp, cop)
        return self.machine_heat_kwh / cop
        
        cop = MACHINE_COP_1[0]
    def get_water_eq(self):
        energy_for_model_1c = 1.0/self.capasity_c_p_kwh
        water_1000l_1c = 1.1666676
        return energy_for_model_1c / water_1000l_1c
        

def optimize_day( config, normal_temp_start, optimized_temp_start, day_loop, model : Model, prices, temps ):
    usage_mask = [1,] * 24
    assert( len(prices) == 24 )
    assert( len(temps) == 24 )

    normal_price, (normal_mask, normal_temp, normal_ele_mach, normal_ele_dir) = model.calculate_day( normal_temp_start, usage_mask, prices, temps )

    while True:
        old_price, _ = model.calculate_day( optimized_temp_start, usage_mask, prices, temps )

        best_action = -1
        best_value = 0
        best_hour = 0
        loop_mask = list(usage_mask)
        for hloop in range(0,24):
            for action in [2]:
                loop_mask[hloop] = action
                try:
                    new_price, _= model.calculate_day( optimized_temp_start, loop_mask, prices, temps )
                except TooLowTemp:
                    continue
                save = old_price - new_price
                if save > best_value:
                    best_value = save
                    best_action = action
                    best_hour = hloop
        if best_action == -1:
            # LOG.info("NO more actions found")
            break 
        
        usage_mask[best_hour] = best_action
        # LOG.info("Doing action %s at %d", best_action, best_hour)
    opt_price, ( opt_mask, opt_temp, opt_ele_mach, opt_ele_dir) = model.calculate_day( optimized_temp_start, usage_mask, prices, temps )

    if config.plots_day:

        LOG.info("Optimization model %s day %03d done - original price: %s - opt price: %s -- orig cons: %0.1f opt cons: %0.1f", model.name, day_loop, normal_price, opt_price, normal_ele_mach, opt_ele_mach )
        fig = pylab.figure(figsize=(12,12))
        ax = pylab.subplot(3,1,1)
        pylab.plot( normal_mask, marker='x',label="Normal usage")
        pylab.plot( opt_mask, marker='x',label="Optimized usage" )
        pylab.legend()
        ax.set_title("Machine use usage %")

        ax = pylab.subplot(3,1,2)
        pylab.plot( normal_temp, label="Normal usage" )
        pylab.plot( opt_temp, label="Optimized usage" )
        pylab.legend()
        ax.set_title("Temperature usage")

        ax = pylab.subplot(3,1,3)
        pylab.plot( prices, label="Ele prices" )
        pylab.plot( temps , label="Outside tmp")
        pylab.legend()
        ax.set_title("Enviroment")
        fig.suptitle("Model: %s" % model.name )
        if config.plots_show:
            pylab.show()
        else:
            pylab.savefig( "/tmp/day-%03d.png" % day_loop )
        pylab.close( fig )

    return (normal_price, opt_price, normal_temp[-1], opt_temp[-1], normal_ele_mach + normal_ele_dir,  opt_ele_mach + opt_ele_dir)


def optimize_model( config, model : Model, prices, temps ):
    day_prices_normal = []
    day_prices_opt = []
    day_ele_normal = []
    day_ele_opt = []

    # 
    # Water equivalent: how many 1000L of water we need to have same cap? 
    # 1000L takes 1163.89 kw to heat 1C
    #
    # to heat 1C model we need => 1.0 = model.capasity_c_p_kwh*x => x = 1.0/model.capasity_c_p_kwh
    # and same water eq A * y = x
    # y = x / A

    LOG.info("== Begin model=%s cap water eq = %0.2f kL heat loss=%0.1f kw (at -20C)  heat pow=%f kw", model.name, model.get_water_eq(), model.heat_loss_kwh_p_c*20, model.machine_heat_kwh )
    if config.days is not None:
        days_todo = config.days
    else:
        days_todo =  range(0,len(prices)//24)

    normal_temp = model.normal_on_c
    optimized_temp = model.normal_on_c
    for day_loop in days_todo:
        if (day_loop+1)*24 > len(prices):
            break
        
        (normal, optimized, normal_temp, optimized_temp, normal_ele, optimized_ele ) = optimize_day( config, normal_temp, optimized_temp, day_loop, model, prices=prices[day_loop*24:(day_loop+1)*24], temps=temps[day_loop*24:(day_loop+1)*24])
        day_prices_normal .append(normal)
        day_prices_opt.append( optimized)
        day_ele_normal.append(normal_ele)
        day_ele_opt.append( optimized_ele)

    sum_normal = sum(day_prices_normal)
    sum_optimized = sum(day_prices_opt)
    sum_ele_normal = sum( day_ele_normal )
    sum_ele_opt = sum( day_ele_opt )

    if len(days_todo)>5 and config.plots_price:
        fig = pylab.figure()
        ax = pylab.subplot(2,1,1)
        pylab.plot(day_prices_normal, label="Prices normal")
        pylab.plot(day_prices_opt, label="Prices opt")
        ax.set_title("Day prices")
        ax = pylab.subplot(2,1,2)
        pylab.plot(day_ele_normal, label="Ele normal")
        pylab.plot(day_ele_opt, label="Ele opt")
        ax.set_title("Day consumption")
        fig.suptitle("Model: %s" % model.name )
    LOG.info("Total prices: %d e %d e => %0.2f %% -- consm normal=%0.1f opt=%0.1f Mwh", sum_normal, sum_optimized, 100*(1.0 - sum_optimized/sum_normal), sum_ele_normal/1000, sum_ele_opt/1000)
    return (sum_optimized, sum_normal)

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, nargs="+")
    parser.add_argument("--plots-day", action="store_true")
    parser.add_argument("--plots-price", action="store_true")
    parser.add_argument("--plots-show", action="store_true")
    parser.add_argument("--plot-data", action="store_true")
    parser.add_argument("--mode", choices=["single","compare","scale"], default="single")
    parser.add_argument("--model", default="270l_8kw")
    return parser.parse_args()

MODELS=["200l_8kw", "270l_8kw","270l_10kw", "500l_8kw", "500l_10kw", "1000l_8kw", "1000l_10kw"]

def get_model( name: str ) -> Model:
    if name == "270l_8kw":
        mod = Model()
    elif name == "270l_10kw":
        mod = Model()
        mod.machine_heat_kwh = 10
    elif name == "200l_8kw":
        mod = Model()
        mod.machine_heat_kwh = 8
        mod.capasity_c_p_kwh = water_equivalent( 360 + 200)
    elif name == "500l_8kw":
        mod = Model()
        mod.capasity_c_p_kwh = water_equivalent( 360 + 500)
    elif name == "500l_10kw":
        mod = Model()
        mod.machine_heat_kwh = 10
        mod.capasity_c_p_kwh = water_equivalent( 360 + 500)
    elif name == "1000l_10kw":
        mod = Model()
        mod.machine_heat_kwh = 10
        mod.capasity_c_p_kwh = water_equivalent( 360 + 1000)
    elif name == "1000l_8kw":
        mod = Model()
        mod.machine_heat_kwh = 8
        mod.capasity_c_p_kwh = water_equivalent( 360 + 1000)
    else:
        raise Exception("Invalid model")
    mod.name = name
    return mod

def main(config):
    prices = load_prices()
    temps = load_temp()

    assert( len(prices) == len(temps ))
    assert(  (len(temps ) % 24 ) == 0 )
    
    LOG.info("Loaded data for %d days", len(temps)//24)
    if config.plot_data:
        x_axis=[x/24.0 for x in range(len(temps))]
        ax = pylab.subplot(2,1,1)
        pylab.plot(x_axis, prices)
        ax.set_title("Prices")
        ax = pylab.subplot(2,1,2)
        pylab.plot( x_axis, temps)
        ax.set_title("Temperatures")
        
    
    if config.mode == "single":
        model = get_model( config.model )
        optimize_model( config, model, prices, temps )
    elif config.mode == "compare":
        for model_name in MODELS:
            model = get_model( model_name )
            optimize_model( config, model, prices, temps )
        pylab.show()
    elif config.mode == "scaling":
        data_y_o = []
        data_y_n = []
        data_x = []
        for loop in range(-4,20):
            model = Model()
            prec = ( 1.0 + loop / 8.0 )
            model.capasity_c_p_kwh /= prec
            v_o, v_n = optimize_model( config, model, prices, temps )
            data_y_o.append(v_o)
            data_y_n.append(v_n)
            data_x.append(prec)
        
        pylab.plot(data_x, data_y_n, marker='x', label="Nominal - inc cap by X")
        pylab.plot(data_x, data_y_o, marker='o', label="Market - inc cap by X")
        
        data_y_o = []
        data_y_n = []
        data_x = []
        for loop in range(-4,+10):
            model = Model()
            prec = ( 1.0 + loop / 8.0 )
            model.increase_power( prec )
            v_o, v_n = optimize_model( config, model, prices, temps )
            data_y_o.append(v_o)
            data_y_n.append(v_n)
            data_x.append(prec)

        pylab.plot(data_x, data_y_n, marker='x', label="Nominal - inc pow by X")
        pylab.plot(data_x, data_y_o, marker='o', label="Market - inc pow by X")


        data_y_o = []
        data_y_n = []
        data_x = []
        for loop in range(-4,+20):
            model = Model()
            prec = ( 1.0 + loop / 8.0 )
            model.capasity_c_p_kwh /= prec
            model.increase_power( 1.25 )
            v = optimize_model( config, model, prices, temps )
            v_o, v_n = optimize_model( config, model, prices, temps )
            data_y_o.append(v_o)
            data_y_n.append(v_n)
            data_x.append(prec)

        pylab.plot(data_x, data_y_n, marker='x', label="Nominal - inc cap by X - p1.25")
        pylab.plot(data_x, data_y_o, marker='o', label="Market - inc cap by X - p1.25")


        data_y_o = []
        data_y_n = []
        data_x = []
        for loop in range(-4,+20):
            model = Model()
            prec = ( 1.0 + loop / 8.0 )
            model.capasity_c_p_kwh /= prec
            model.increase_power( 0.75 )
            v = optimize_model( config, model, prices, temps )
            v_o, v_n = optimize_model( config, model, prices, temps )
            data_y_o.append(v_o)
            data_y_n.append(v_n)
            data_x.append(prec)


        pylab.plot(data_x, data_y_n, marker='x', label="Nominal - inc cap by X - p0.75")
        pylab.plot(data_x, data_y_o, marker='o', label="Market - inc cap by X - p0.75")

        pylab.legend()
        pylab.show()
    else:
        raise Exception ("Invalid mode")

if __name__=="__main__":
    logging.basicConfig( level=logging.INFO)
    main(get_config())