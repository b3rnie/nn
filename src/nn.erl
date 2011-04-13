%%-*- erlang-indent-level: 2 -*-
%% Copied from http://www.trapexit.org/Erlang_and_Neural_Networks
-module(nn).

-export([ perceptron/3
        , sigmoid/1
        , dot_prod/2
        , feed_forward/2
        , replace_input/2
        , convert_to_list/1
        , connect/2
        ]).


sigmoid(N) ->
  1 / (1 + math:exp(-N)).

sigmoid_deriv(N) ->
  math:exp(-N) / (1 + math:exp(-2 * N)).

dot_prod(L1, L2)
  when length(L1) =:= length(L2) ->
  dot_prod(L1, L2, 0).

dot_prod([], [], Acc) ->
  Acc;
dot_prod([H1|T1], [H2|T2], Acc) ->
  dot_prod(T1, T2, Acc + H1*H2).


feed_forward(Weights, Inputs) ->
  sigmoid(dot_prod(Weights, Inputs)).

feed_forward_deriv(Weights, Inputs) ->
  sigmoid_deriv(dot_prod(Weights, Inputs)).

perceptron(Weights, Inputs, OutputPids) ->
  receive
    {stimulate, Input} ->
      %% add Input to Inputs to get New_Inputs...
      NewInputs = replace_input(Inputs, Input),
      
      %% calculate output of perceptron...
      Output = feed_forward(Weights, convert_to_list(NewInputs)),
      
      %% stimulate the perceptron my output is connected to
      case OutputPids of
        [] -> io:format("~p outputs: ~p~n", [self(), Output]);
        _  ->
          lists:foreach(fun(OutputPid) ->
                            OutputPid ! {stimulate, {self(), Output}}
                        end, OutputPids)
            end,
      perceptron(Weights, NewInputs, OutputPids);
    {connect_to_output, ReceiverPid} ->
      CombinedOutput = [ReceiverPid|OutputPids],
      io:format("~p output connected to ~p: ~p~n", [self(), ReceiverPid,
                                                    CombinedOutput]),
      perceptron(Weights, Inputs, CombinedOutput);
    {connect_to_input, SenderPid} ->
      CombinedInput = [{SenderPid, 0.5}|Inputs],
      io:format("~p inputs connected to ~p: ~p~n", [self(), SenderPid,
                                                    CombinedInput]),
      perceptron([0.5|Weights], CombinedInput, OutputPids);
    {pass, InputValue} ->
      lists:foreach(fun(OutputPid) ->
                        io:format("stimulating ~p with ~p~n", [OutputPid,
                                                               InputValue]),
                        OutputPid ! {stimulate, {self(), InputValue}}
                    end, OutputPids),
      perceptron(Weights, Inputs, OutputPids)
  end.

connect(SenderPid, ReceiverPid) ->
  SenderPid ! {connect_to_output, ReceiverPid},
  ReceiverPid ! {connect_to_input, SenderPid}.


replace_input(Inputs, Input) ->
  {InputPid, _} = Input,
  lists:keyreplace(InputPid, 1, Inputs, Input).

convert_to_list(Inputs) ->
  lists:map(fun({_, Val}) ->
                Val
            end, Inputs).


%% 4> Pid = spawn(nn, perceptron, [[0.5, 0.2], [{1,0.6}, {2,0.9}], []]).
%% <0.48.0>
%% 5> Pid ! {stimulate, {1, 0.3}}.
%% {stimulate,{1,0.3}}
%% <0.48.0> outputs: 0.5817593768418363

